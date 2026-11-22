"""Night-time fog prediction for one point and time.

  python scripts/predict.py --time "2024-12-15 03:00" --lon 32.99 --lat 40.13
  python scripts/predict.py --time "2024-12-15 03:00" --lon 32.99 --lat 40.13 --no-era5
  python scripts/predict.py --example fog        (offline demo, no internet needed)
"""
import argparse
import sys

sys.path.insert(0, "src")
from turkey_fog.config import load_config
from turkey_fog.predict import predict_example, predict_point


def show_example(kind):
    ex, results = predict_example(f"examples/{kind}.json")
    print(f"Example: {ex['station']} {ex['time']} UTC (lon {ex['lon']}, lat {ex['lat']})")
    print(f"METAR report at that time: {'FOG' if ex['metar_label'] else 'no fog'}")
    print(ex["note"])
    print()
    for r in results:
        print(f"  {r['model']:10s} fog probability {r['probability']:.2f} (threshold {r['threshold']:.2f})"
              f" -> {'FOG' if r['fog'] else 'no fog'}")


def show_point(cfg, args):
    r = predict_point(cfg, args.lon, args.lat, args.time, use_era5=not args.no_era5, timeout=args.era5_timeout)
    for n in r["notes"]:
        print(n)
    print(f"Time: {r['time']} UTC   Point: lon {r['lon']}, lat {r['lat']}")
    print(f"SEVIRI scan: {r['product']}")
    print(f"Model: {r['model']}   fog probability {r['probability']:.2f} (threshold {r['threshold']:.2f})")
    print("Result:", "FOG" if r["fog"] else "no fog")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Night-time fog prediction from SEVIRI (+ ERA5-Land)")
    parser.add_argument("--example", choices=["fog", "clear"], help="run a saved offline example")
    parser.add_argument("--time", help='UTC time, e.g. "2024-12-15 03:00"')
    parser.add_argument("--lon", type=float)
    parser.add_argument("--lat", type=float)
    parser.add_argument("--no-era5", action="store_true", help="use the SEVIRI-only model")
    parser.add_argument("--era5-timeout", type=int, default=None, help="seconds to wait for ERA5 (default in config)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    if args.example:
        show_example(args.example)
    elif args.time and args.lon is not None and args.lat is not None:
        try:
            show_point(load_config(args.config), args)
        except (ValueError, LookupError) as e:
            print("No prediction:", e)
            sys.exit(1)
    else:
        parser.error("give --example, or --time, --lon and --lat")







