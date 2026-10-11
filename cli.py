#!/usr/bin/env python3
"""
Command-line interface for time series analysis.

This module provides a CLI for running time series analysis
without the web interface.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.app import TimeSeriesAnalysisApp


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Time Series Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python cli.py
  
  # Run with custom config file
  python cli.py --config custom_config.yaml
  
  # Run with verbose output
  python cli.py --verbose
  
  # Run specific components only
  python cli.py --skip-anomaly-detection --skip-visualization
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--skip-anomaly-detection',
        action='store_true',
        help='Skip anomaly detection'
    )
    
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--skip-model-save',
        action='store_true',
        help='Skip saving trained models'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Check if config file exists
        if not Path(args.config).exists():
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        
        # Initialize app
        logger.info("Initializing Time Series Analysis App")
        app = TimeSeriesAnalysisApp(config_path=args.config)
        
        # Update configuration based on CLI arguments
        if args.skip_anomaly_detection:
            app.config['anomaly_detection']['enabled'] = False
            logger.info("Anomaly detection disabled")
        
        if args.skip_model_save:
            app.config['output']['save_models'] = False
            logger.info("Model saving disabled")
        
        # Update output directory
        app.config['output']['results_dir'] = args.output_dir
        
        # Run analysis
        logger.info("Starting time series analysis")
        app.run_complete_analysis()
        
        # Display summary
        summary = app.get_summary()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Data Points: {summary['data_points']}")
        print(f"Training Points: {summary['train_points']}")
        print(f"Test Points: {summary['test_points']}")
        print(f"Models Trained: {', '.join(summary['models_trained'])}")
        print(f"Forecast Periods: {summary['forecast_periods']}")
        
        if 'best_model' in summary:
            print(f"Best Model: {summary['best_model']}")
        
        if 'metrics' in summary:
            print("\nModel Performance:")
            for model, metrics in summary['metrics'].items():
                print(f"  {model}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")
        
        if 'anomalies_detected' in summary:
            print("\nAnomaly Detection:")
            for method, count in summary['anomalies_detected'].items():
                print(f"  {method.title()}: {count} anomalies")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*60)
        
        logger.info("Analysis completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
