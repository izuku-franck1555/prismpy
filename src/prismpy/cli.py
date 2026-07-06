#!/usr/bin/env python
"""
prismpy CLI - Command-line interface for data-to-model translation.

This module provides the main entry point for the prismpy framework,
supporting configuration loading, pipeline execution, and output validation.

Usage:
    prismpy translate --config project_config.yaml
    prismpy validate --platform sarra_py --output-dir outputs/
    prismpy info --config project_config.yaml

Examples:
    # Run full translation pipeline
    prismpy translate --config mali_maize.yaml --targets sarra_py craft

    # Validate existing outputs
    prismpy validate --platform acea --output-dir outputs/acea/

    # Show configuration info
    prismpy info --config mali_maize.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from prismpy.config.loader import (
    load_config, save_config, load_dome_config, load_auto_config, load_raw_yaml
)
from prismpy.config.schema import Platform, ProjectConfig
from prismpy.standards import (
    IcasaValidator, AceConverter, validate_config, export_ace, import_ace
)
from prismpy.pipeline.executor import TranslationPipeline, PipelineStage
from prismpy.validators import (
    SarraPyValidator,
    CraftValidator,
    PythiaValidator,
    AceaValidator,
    ValidationResult,
)


# Configure logging
def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity settings.

    Args:
        verbose: Enable debug logging
        quiet: Suppress info logging (errors only)
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_validator_for_platform(platform: str):
    """Get the appropriate validator for a platform.

    Args:
        platform: Platform name (sarra_py, craft, pythia, acea)

    Returns:
        Validator class
    """
    validators = {
        "sarra_py": SarraPyValidator,
        "craft": CraftValidator,
        "pythia": PythiaValidator,
        "acea": AceaValidator,
    }

    platform_lower = platform.lower().replace("-", "_")
    if platform_lower not in validators:
        raise ValueError(f"Unknown platform: {platform}. "
                        f"Available: {list(validators.keys())}")

    return validators[platform_lower]


def cmd_translate(args: argparse.Namespace) -> int:
    """Execute translation pipeline.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger = logging.getLogger(__name__)

    # Load configuration - support both legacy and DOME formats
    try:
        if args.base and args.dome:
            # DOME format: base + dome
            merged, _ = load_dome_config(args.base, args.dome)
            config = ProjectConfig.model_validate(merged)
            logger.info(f"Loaded DOME config: base={args.base}, dome={args.dome}")
        elif args.base:
            # Base only (no DOME) - use auto-load
            raw = load_auto_config(args.base)
            config = ProjectConfig.model_validate(raw)
            logger.info(f"Loaded base config: {args.base}")
        else:
            # Legacy format
            config = load_config(args.config)
            logger.info(f"Loaded configuration: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Override targets if specified
    if args.targets:
        targets = []
        for t in args.targets:
            try:
                platform = Platform(t.lower().replace("-", "_"))
                targets.append(platform)
            except ValueError:
                logger.error(f"Unknown target platform: {t}")
                return 1
        config.targets = targets
        try:
            config.assert_craft_resolution_compatible()
        except ValueError as e:
            logger.error(str(e))
            return 1
        logger.info(f"Target platforms: {[t.value for t in targets]}")

    # Create and run pipeline
    try:
        pipeline = TranslationPipeline(config)

        logger.info("=" * 60)
        logger.info(f"Starting translation: {config.project.name}")
        logger.info(f"Region: {config.region.name}")
        logger.info(f"Crop: {config.crop.name}")
        logger.info("=" * 60)

        # Run pipeline stages
        if args.stage:
            stage = PipelineStage(args.stage)
            logger.info(f"Running single stage: {stage.value}")
            result = pipeline.execute(stages=[stage])
        else:
            logger.info("Running full pipeline...")
            result = pipeline.execute()

        # Report results
        if result.success:
            logger.info("=" * 60)
            logger.info("Translation completed successfully!")
            logger.info(f"Output directory: {config.output.base_dir}")
            logger.info("=" * 60)
            return 0
        else:
            logger.error("=" * 60)
            logger.error("Translation failed!")
            # Extract errors from stage results
            for stage_name, stage_result in result.stages.items():
                if stage_result.errors:
                    for error in stage_result.errors:
                        logger.error(f"  [{stage_name}] {error}")
            logger.error("=" * 60)
            return 1

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_generate_scenario_set(args: argparse.Namespace) -> int:
    """Generate ISIMIP3b projection packages for a scenario set."""
    logger = logging.getLogger(__name__)
    from prismpy.packaging.scenario_set_generator import generate_scenario_set

    try:
        spec = load_raw_yaml(args.config)
    except Exception as e:  # noqa: BLE001 — surface any load error to the CLI
        logger.error(f"Failed to load scenario-set config: {e}")
        return 1

    try:
        baseline_config = load_config(spec["baseline_config"])
        result = generate_scenario_set(
            baseline_package=spec["baseline_package"],
            baseline_config=baseline_config,
            aoi_bbox=spec["aoi_bbox"],
            gcms=spec["gcms"],
            ssps=spec["ssps"],
            time_slices=[tuple(s) for s in spec["time_slices"]],
            region_name=spec["region_name"],
            crop_name=spec["crop_name"],
            output_dir=spec["output_dir"],
            cache_dir=Path(spec["cache_dir"]) if spec.get("cache_dir") else None,
        )
    except KeyError as e:
        logger.error(f"Scenario-set config missing required key: {e}")
        return 1
    except Exception as e:  # noqa: BLE001 — report generation failure, exit non-zero
        logger.error(f"Scenario-set generation failed: {e}")
        return 1

    print(f"Baseline: {result.baseline_package}")
    print(f"Generated {len(result.projection_packages)} projection package(s):")
    for package, (gcm, ssp, time_slice) in zip(
        result.projection_packages, result.matrix
    ):
        print(f"  [{gcm} {ssp} {time_slice[0]}-{time_slice[1]}] {package}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate platform outputs.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for valid, 1 for invalid)
    """
    logger = logging.getLogger(__name__)

    try:
        # Get validator
        validator_class = get_validator_for_platform(args.platform)
        validator = validator_class(args.output_dir)

        logger.info(f"Validating {args.platform} outputs in {args.output_dir}")

        # Run validation
        result = validator.validate()

        # Print results
        print()
        print("=" * 60)
        print(result.summary())
        print("=" * 60)

        if result.issues:
            print()
            print("Issues found:")
            for issue in result.issues:
                print(f"  {issue}")

        print()
        if result.valid:
            print("RESULT: VALID")
            return 0
        else:
            print(f"RESULT: INVALID ({result.n_errors} errors)")
            return 1

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Display configuration information.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    logger = logging.getLogger(__name__)

    try:
        config = load_config(args.config)

        print()
        print("=" * 60)
        print("prismpy Configuration")
        print("=" * 60)
        print()
        print(f"Project Name: {config.project.name}")
        print(f"Description:  {config.project.description or 'N/A'}")
        print()
        print("Region:")
        print(f"  Name:    {config.region.name}")
        print(f"  Country: {config.region.country}")
        if config.region.bounds:
            bounds = config.region.bounds
            print(f"  Bounds:  [{bounds.minx}, {bounds.miny}, {bounds.maxx}, {bounds.maxy}]")
        print()
        print("Crop:")
        print(f"  Name:    {config.crop.name}")
        print(f"  Variety: {config.crop.variety or 'N/A'}")
        if config.crop.calendar:
            print(f"  Planting DOY: {config.crop.calendar.planting_doy}")
            print(f"  Harvest DOY:  {config.crop.calendar.harvest_doy}")
        print()
        print("Temporal:")
        print(f"  Start Year: {config.temporal.start_year}")
        print(f"  End Year:   {config.temporal.end_year}")
        print(f"  Spinup:     {config.temporal.spinup_years} years")
        print()
        print("Targets:")
        for target in config.targets:
            print(f"  - {target.value}")
        print()
        print("Output:")
        print(f"  Directory: {config.output.base_dir}")
        print(f"  Structure: {config.output.structure}")
        print()
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new configuration file.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    logger = logging.getLogger(__name__)

    output_path = Path(args.output or "project_config.yaml")

    if output_path.exists() and not args.force:
        logger.error(f"File already exists: {output_path}")
        logger.info("Use --force to overwrite")
        return 1

    # Create default configuration
    from prismpy.config.defaults import DEFAULT_CONFIG

    try:
        # Write example config
        save_config(DEFAULT_CONFIG, output_path)
        logger.info(f"Created configuration template: {output_path}")
        logger.info("Edit the file to customize for your project.")
        return 0

    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        return 1


def cmd_validate_icasa(args: argparse.Namespace) -> int:
    """Validate a config file against ICASA standards.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for valid, 1 for invalid)
    """
    logger = logging.getLogger(__name__)

    try:
        # Load config
        config = load_raw_yaml(args.config)
        logger.info(f"Loaded configuration: {args.config}")

        # Validate
        validator = IcasaValidator()
        result = validator.validate(config, strict=args.strict)

        # Print results
        print()
        print("=" * 60)
        print("ICASA Validation Results")
        print("=" * 60)
        print()

        if result.errors:
            print(f"ERRORS ({len(result.errors)}):")
            for error in result.errors:
                print(f"  [ERROR] {error}")
            print()

        if result.warnings:
            print(f"WARNINGS ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  [WARN] {warning}")
            print()

        if result.info:
            print(f"INFO ({len(result.info)}):")
            for info in result.info:
                print(f"  [INFO] {info}")
            print()

        print("=" * 60)
        if result.is_valid:
            print("RESULT: VALID (ICASA-compliant)")
            return 0
        else:
            print(f"RESULT: INVALID ({len(result.errors)} errors)")
            return 1

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_export_ace(args: argparse.Namespace) -> int:
    """Export config to AgMIP ACE JSON format.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    logger = logging.getLogger(__name__)

    try:
        # Load config
        config = load_raw_yaml(args.config)
        logger.info(f"Loaded configuration: {args.config}")

        # Convert to ACE
        converter = AceConverter()
        ace_data = converter.export_ace(config)

        # Write output
        output_path = Path(args.output)
        converter.export_ace_file(config, output_path)

        logger.info(f"Exported ACE JSON to: {output_path}")
        print(f"Successfully exported to: {output_path}")

        # Print summary
        if 'experiments' in ace_data:
            print(f"  Experiments: {len(ace_data['experiments'])}")
        if 'soils' in ace_data:
            print(f"  Soils: {len(ace_data['soils'])}")

        return 0

    except Exception as e:
        logger.error(f"Export failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_import_ace(args: argparse.Namespace) -> int:
    """Import AgMIP ACE JSON to prismpy config.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    logger = logging.getLogger(__name__)

    try:
        # Import ACE
        converter = AceConverter()
        config = converter.import_ace_file(args.ace)

        logger.info(f"Imported ACE JSON from: {args.ace}")

        # Write output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import yaml
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"Saved config to: {output_path}")
        print(f"Successfully imported ACE to: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Import failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_migrate(args: argparse.Namespace) -> int:
    """Migrate legacy config to base + DOME format.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    logger = logging.getLogger(__name__)

    try:
        from prismpy.config.dome_merger import DomeMerger

        # Load legacy config
        legacy = load_raw_yaml(args.legacy)
        logger.info(f"Loaded legacy config: {args.legacy}")

        # Extract base and DOME
        merger = DomeMerger()
        base = merger.extract_base(legacy)
        dome = merger.extract_dome(legacy, args.platform)

        # Add ICASA format marker to base
        base['_meta'] = base.get('_meta', {})
        base['_meta']['format'] = 'icasa_ace'
        base['_meta']['migrated_from'] = str(args.legacy)

        # Write outputs
        import yaml

        base_path = Path(args.output_base)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        with open(base_path, 'w', encoding='utf-8') as f:
            yaml.dump(base, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Saved base config to: {base_path}")

        dome_path = Path(args.output_dome)
        dome_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dome_path, 'w', encoding='utf-8') as f:
            yaml.dump(dome, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Saved DOME to: {dome_path}")

        print(f"Migration complete!")
        print(f"  Base config: {base_path}")
        print(f"  Platform DOME: {dome_path}")
        print()
        print("Next steps:")
        print(f"  1. Review and edit {base_path} (ICASA-compliant agronomic data)")
        print(f"  2. Review and edit {dome_path} (platform-specific settings)")
        print(f"  3. Run: prismpy translate --base {base_path} --dome {dome_path}")

        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="prismpy",
        description="Data-to-model translation framework for crop modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  prismpy translate --config project.yaml
  prismpy validate --platform acea --output-dir outputs/
  prismpy info --config project.yaml
  prismpy init --output my_project.yaml

For more information, see: https://github.com/your-repo/prismpy
        """,
    )

    # Global options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands",
    )

    # translate command
    translate_parser = subparsers.add_parser(
        "translate",
        help="Run translation pipeline",
        description="Execute the full data-to-model translation pipeline",
    )
    # Legacy format (single file)
    translate_parser.add_argument(
        "-c", "--config",
        help="Path to project configuration YAML file (legacy format)",
    )
    # DOME format (base + dome)
    translate_parser.add_argument(
        "-b", "--base",
        help="Path to ICASA-compliant base configuration (DOME format)",
    )
    translate_parser.add_argument(
        "-d", "--dome",
        help="Path to platform DOME overlay (DOME format)",
    )
    translate_parser.add_argument(
        "-t", "--targets",
        nargs="+",
        help="Target platforms (overrides config): sarra_py, craft, pythia, acea",
    )
    translate_parser.add_argument(
        "-s", "--stage",
        choices=["retrieve", "harmonize", "translate", "validate", "package"],
        help="Run only a specific pipeline stage",
    )
    translate_parser.set_defaults(func=cmd_translate)

    # generate-scenario-set command
    scenario_set_parser = subparsers.add_parser(
        "generate-scenario-set",
        help="Generate ISIMIP3b projection packages for a scenario set",
        description=(
            "Clone-and-swap a baseline package into ISIMIP3b projection "
            "packages across a (GCM x SSP x time-slice) matrix."
        ),
    )
    scenario_set_parser.add_argument(
        "-c", "--config",
        required=True,
        help=(
            "Path to the scenario-set YAML (keys: baseline_package, "
            "baseline_config, aoi_bbox, gcms, ssps, time_slices, region_name, "
            "crop_name, output_dir; optional cache_dir)"
        ),
    )
    scenario_set_parser.set_defaults(func=cmd_generate_scenario_set)

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate platform outputs",
        description="Validate generated outputs for a specific platform",
    )
    validate_parser.add_argument(
        "-p", "--platform",
        required=True,
        choices=["sarra_py", "craft", "pythia", "acea"],
        help="Platform to validate",
    )
    validate_parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Path to output directory to validate",
    )
    validate_parser.set_defaults(func=cmd_validate)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Display configuration information",
        description="Show details of a project configuration file",
    )
    info_parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to project configuration YAML file",
    )
    info_parser.set_defaults(func=cmd_info)

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new configuration file",
        description="Create a template configuration file",
    )
    init_parser.add_argument(
        "-o", "--output",
        help="Output path for configuration file (default: project_config.yaml)",
    )
    init_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing file",
    )
    init_parser.set_defaults(func=cmd_init)

    # validate-icasa command (NEW - DOME support)
    validate_icasa_parser = subparsers.add_parser(
        "validate-icasa",
        help="Validate config against ICASA standards",
        description="Validate a base configuration against ICASA standards",
    )
    validate_icasa_parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to configuration file to validate",
    )
    validate_icasa_parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation (all fields required)",
    )
    validate_icasa_parser.set_defaults(func=cmd_validate_icasa)

    # export-ace command (NEW - DOME support)
    export_ace_parser = subparsers.add_parser(
        "export-ace",
        help="Export config to AgMIP ACE JSON format",
        description="Export a configuration to AgMIP-compatible ACE JSON format",
    )
    export_ace_parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to base configuration file",
    )
    export_ace_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output path for ACE JSON file",
    )
    export_ace_parser.set_defaults(func=cmd_export_ace)

    # import-ace command (NEW - DOME support)
    import_ace_parser = subparsers.add_parser(
        "import-ace",
        help="Import AgMIP ACE JSON to config",
        description="Import AgMIP ACE JSON to prismpy configuration format",
    )
    import_ace_parser.add_argument(
        "-a", "--ace",
        required=True,
        help="Path to ACE JSON file to import",
    )
    import_ace_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output path for configuration YAML file",
    )
    import_ace_parser.set_defaults(func=cmd_import_ace)

    # migrate command (NEW - DOME support)
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate legacy config to base + DOME format",
        description="Split a legacy single-file config into ICASA base + platform DOME",
    )
    migrate_parser.add_argument(
        "-l", "--legacy",
        required=True,
        help="Path to legacy configuration file",
    )
    migrate_parser.add_argument(
        "-p", "--platform",
        required=True,
        choices=["craft", "pythia", "acea", "sarra_py"],
        help="Target platform for the DOME",
    )
    migrate_parser.add_argument(
        "--output-base",
        required=True,
        help="Output path for base configuration (ICASA-compliant)",
    )
    migrate_parser.add_argument(
        "--output-dome",
        required=True,
        help="Output path for platform DOME",
    )
    migrate_parser.set_defaults(func=cmd_migrate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging
    setup_logging(
        verbose=getattr(args, 'verbose', False),
        quiet=getattr(args, 'quiet', False),
    )

    # Run command
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
