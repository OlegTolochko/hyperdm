import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/era5_t2m.grib")
    parser.add_argument("--output", default="data/era5_t2m.npy")
    parser.add_argument("--north", type=float, default=83)
    parser.add_argument("--west", type=float, default=-169)
    parser.add_argument("--south", type=float, default=7)
    parser.add_argument("--east", type=float, default=-35)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.input)
    t2m = ds["t2m"].sel(latitude=slice(args.north, args.south),
                        longitude=slice(args.west, args.east))
    values = t2m.astype("float32").load().values
    np.save(output, values)

    metadata = {
        "source": args.input,
        "variable": "2m_temperature",
        "shape": list(values.shape),
        "bounds": {
            "north": args.north,
            "west": args.west,
            "south": args.south,
            "east": args.east,
        },
        "time_start": str(t2m.time.values[0]),
        "time_end": str(t2m.time.values[-1]),
        "units": "K",
        "normalization": "[210, 313] K maps to [-1, 1] in data/era5.py",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {output} with shape {values.shape}")
    print(f"Kelvin range: {float(values.min()):.3f} to {float(values.max()):.3f}")


if __name__ == "__main__":
    main()
