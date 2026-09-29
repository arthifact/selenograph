"""One-time, offline publication of preserved maps as ordinary processed site sections.

    uv run --no-project python -B -m app.prepare_reference_section --help
    uv run --no-project python -B -m app.prepare_reference_section --dry-run

Importing this module does nothing. ``inventory`` validates without writing;
``run`` publishes each site to processed_root/<site_id>(reference), alongside
public-region folders directly under processed_root. --site selects one source
site; the default is all sites. Neither API writes user paintings, fits models,
or downloads.
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from app import dataset_utils as data
from app import paths

EXPECTED_REFERENCE_ARTIFACTS = 88
LABEL_STATUS = ("User-marked verified; preserved unreviewed historical snapshot; "
                "not independent certification")
UNVERIFIED_STATUS = "Preserved unreviewed historical snapshot; not independent certification"
README = """# {section} (restricted, local use only)

Each site is a normal processed section with its own working_dems.json. The
default layout has NO public or reference wrapper directories:

    data/processed_data/
      manifest.json
      leibnitz_beta_plateau/working_dems.json
      nobile_rim_1/working_dems.json
      nobile_rim_2/working_dems.json
      ... other public-region folders ...
      mons-mouton(reference)/working_dems.json
      nobile1(reference)/working_dems.json
      nobile2(reference)/working_dems.json
      nobile1-ms1(reference)/working_dems.json

This folder's section is {section}; collection is provenance, not a UI section.
The importer can select one source site with --site; by default it publishes all.
This is a normal processed dataset, discovered through working_dems.json. The app
needs only the processed folder, not a live professor-map importer. Original
sources remain in data/reference_data; source_refs are provenance, not download
instructions. All source restrictions continue to apply.

IMPORTANT: raster files here are HARD LINKS to immutable preserved sources.
Writing either pathname changes the same underlying file. Never edit, chmod,
reproject, or overwrite these raster files in place. The importer only reads
sources and creates links; it does not change source contents or permissions.
App painting edits belong exclusively in the ordinary output maps/drafts folders,
not here. These seed maps are editable through that normal output-copy behavior;
there is no special app lock or live reference lookup.

For portability, an authorized offline copy of the processed tree can materialize
hard links as ordinary files later. Keep the public-region folders alongside this
one: feature_sources use paths such as ../nobile_rim_1/tiles/... to matching
WHOLE public DEM tiles. The
six model features are computed there by generic core.build and transported to
the preserved DEM grid; the original reference DEM supplies display geometry.
NAC and radar companions remain display layers, not substitute model features.

One snapshot per tile: saved if present, otherwise draft. Accepted-prediction
flags are the union of all observed snapshots. Absent confidence remains unknown;
no confidence values are synthesized. The default annotations.verified=true is
an explicit USER REQUEST, not inferred confidence, independent certification, or
a claim that historical snapshots received a new scientific review. --no-verified
opts out. The manifest and dataset.json record this distinction and all hashes.

Re-running is validation-only when the existing manifest and materialized hashes
match exactly. Other nonempty destinations are refused, never repaired in place.
All selected plans and existing destinations are validated before any writing;
all new sections are staged before publication. Each site folder is renamed
atomically. An interruption between renames may leave complete sites published;
re-running validates those sites and resumes the remaining ones.
"""


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _npy_bytes(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Unsafe tile/layer identifier: {value!r}")
    return value


def _roots(processed_root, reference_root):
    processed = Path(processed_root or paths.DEM_DIR).resolve()
    reference = Path(reference_root or paths.PROFESSOR_MAPS_DIR).resolve()
    if reference.is_relative_to(processed) or processed.is_relative_to(reference):
        raise ValueError("Source reference and processed roots must not overlap")
    if not processed.is_dir() or not reference.is_dir():
        raise ValueError("Existing processed and preserved reference roots are required")
    return processed, reference


def _catalogs(processed, targets):
    """Reserve the generic map-ID namespace, including unrelated public regions."""
    catalogs, stems = {}, set()
    for directory, children, files in os.walk(processed, followlinks=False):
        children[:] = sorted(name for name in children if not name.startswith(".")
                             and Path(directory) / name not in targets)
        if "working_dems.json" not in files:
            continue
        manifest = data._path(processed, Path(directory) / "working_dems.json")
        records = data._json(manifest)
        if not isinstance(records, list):
            raise TypeError(f"Invalid working DEM catalog: {manifest}")
        catalogs[manifest] = records
        for record in records:
            dem = data._path(processed, manifest.parent / record["working_dem"])
            stems.add(dem.stem)
    # Flat legacy maps share the same output-painting namespace as manifest maps.
    stems.update(p.stem for p in processed.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    return catalogs, stems


def _feature_sources(processed, refs, catalogs, hashes):
    """The same region, SfS grid intersections and tile order as reference_training."""
    public = processed
    manifest_path = data._path(public, "manifest.json")
    manifest = data._json(manifest_path)
    hashes.file(manifest_path, "processed/manifest.json")
    result = {ref["record"]["tile_id"]: [] for ref in refs}
    occupied = {ref["record"]["tile_id"]: np.zeros(ref["grid"].shape, bool) for ref in refs}
    for region in sorted({data.REGIONS[r["record"]["site_id"]] for r in refs}):
        folder = data._path(public, region)
        meta_path = data._path(folder, f"metadata/{region}_metadata.json")
        hashes.file(meta_path, f"processed/{region}/metadata/{region}_metadata.json",
                    manifest["regions"][region]["metadata_sha256"])
        metadata = data._json(meta_path)
        catalog = data._path(folder, "working_dems.json")
        hashes.file(catalog, f"processed/{region}/working_dems.json",
                    metadata["outputs"]["working_dems.json"]["checksum"]["sha256"])
        records = catalogs[catalog]
        ids = [record["tile_id"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate public tile IDs: {region}")
        targets = [r for r in refs if data.REGIONS[r["record"]["site_id"]] == region]
        for record in sorted(records, key=lambda r: r["tile_id"]):
            if record.get("collection") != "public" or record.get("site") != region:
                raise ValueError(f"Nonpublic/inconsistent region record: {record['tile_id']}")
            dem = data._path(folder, record["layers"]["sfs"])
            if data._path(folder, record["working_dem"]) != dem:
                raise ValueError("Public working_dem must be its whole SfS feature tile")
            with rasterio.open(dem) as src:
                grid = data._Grid.read(src)
            hits = [r for r in targets if data._intersects(grid, r["grid"])]
            if not hits:
                continue
            data._metric_grid(grid)
            if min(grid.shape) < 2 or max(grid.shape) > 1024:
                raise ValueError("Public feature context must be a whole 2..1024 pixel tile")
            for role in ("sfs", "valid_data", "sfs_support"):
                relative = record["layers"][role]
                hashes.file(data._path(folder, relative), f"processed/{region}/{relative}",
                            metadata["outputs"][relative]["checksum"]["sha256"])
            for ref in hits:
                tile_id = ref["record"]["tile_id"]
                footprint = data._warp(np.ones(grid.shape, np.float32), grid, ref["grid"]) == 1
                data._claim({"record": ref["record"], "occupied": occupied[tile_id]},
                            footprint, record["tile_id"])
                result[tile_id].append(dem)
    for tile_id, sources in result.items():
        if not sources:
            raise ValueError(f"No matching public feature tiles for {tile_id}")
    return result


@dataclass
class _Plan:
    target: Path
    records: list
    provenance: dict
    links: dict
    small_files: dict
    input_hashes: dict


def _plans(processed_root, reference_root, verified, site):
    if not isinstance(verified, bool):
        raise TypeError("verified must be an explicit boolean")
    processed, reference = _roots(processed_root, reference_root)
    hashes = data._Hashes()
    data._path(reference, "dataset.json")
    refs, count = data._references(reference, hashes)
    if count != EXPECTED_REFERENCE_ARTIFACTS:
        raise ValueError(f"Expected {EXPECTED_REFERENCE_ARTIFACTS} reference artifacts, validated {count}")
    if not refs:
        raise ValueError("No reference tiles to publish")
    original = data._json(data._path(reference, "dataset.json"))
    tile_paths = {item["tile_id"]: item["path"] for item in original["tiles"]}
    sites = {}
    for item in original.get("sites", []):
        path = data._path(reference, item["path"])
        hashes.file(path, "reference/" + item["path"])
        sites[item["site_id"]] = data._json(path)
    ids = [_identifier(ref["record"]["tile_id"]) for ref in refs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Reference map ID collision: {ids}")
    available = {_identifier(ref["record"]["site_id"]) for ref in refs}
    if site is not None:
        if _identifier(site) not in available:
            raise ValueError(f"Unknown source site {site!r}; choose from {sorted(available)}")
        refs = [ref for ref in refs if ref["record"]["site_id"] == site]
    selected = sorted({ref["record"]["site_id"] for ref in refs})
    targets = {name: processed / f"{name}(reference)" for name in selected}
    for target in targets.values():
        if target.is_symlink():
            raise ValueError(f"Output must not be a symlink: {target}")
        if target.exists() and not target.is_dir():
            raise ValueError(f"Output is not a directory: {target}")
    catalogs, stems = _catalogs(processed, set(targets.values()))
    collisions = stems.intersection(ref["record"]["tile_id"] for ref in refs)
    if collisions:
        raise ValueError(f"Reference/public map ID collision: {sorted(collisions)}")
    plans = []
    for name in selected:
        # Per-site provenance must be identical whether --site or all sites are used.
        site_hashes = data._Hashes()
        site_hashes.cache = dict(hashes.cache)
        site_hashes.records = dict(hashes.records)
        site_refs = [ref for ref in refs if ref["record"]["site_id"] == name]
        sources = _feature_sources(processed, site_refs, catalogs, site_hashes)
        plans.append(_site_plan(reference, targets[name], site_refs, sources, sites,
                                tile_paths, site_hashes, count, verified))
    return plans


def _site_plan(reference, target, refs, sources, sites, tile_paths, hashes, count, verified):
    records, links, small_files, materialized = [], {}, {}, {}
    status = LABEL_STATUS if verified else UNVERIFIED_STATUS

    def link(relative, source, expected):
        if relative in materialized:
            raise ValueError(f"Duplicate materialized path: {relative}")
        path = data._path(reference, source)
        digest = hashes.file(path, "reference/" + source, expected)
        links[relative] = path
        materialized[relative] = {"sha256": digest, "storage": "hardlink",
                                  "source": "reference/" + source}
        return relative

    def array(relative, values):
        content = _npy_bytes(values)
        small_files[relative] = content
        materialized[relative] = {"sha256": hashlib.sha256(content).hexdigest(),
                                  "storage": "generated_cell_array"}
        return relative

    for ref in sorted(refs, key=lambda r: r["record"]["tile_id"]):
        record, grid = ref["record"], ref["grid"]
        tile_id, site = record["tile_id"], record["site_id"]
        if site not in data.GROUPS:
            raise ValueError(f"No canonical geographic group for {site}")
        tile_path = tile_paths[tile_id]
        tile_sha = hashes.file(data._path(reference, tile_path), "reference/" + tile_path)
        prefix = f"tiles/{tile_id}/{tile_id}"
        layers = {}
        for role, layer in sorted(record["layers"].items()):
            if not layer.get("path"):
                continue
            role = _identifier({"radar-cpr": "cpr", "radar-s1": "radar_s1"}.get(role, role))
            relative = prefix + (".tif" if role == "dem" else f"_{role}.tif")
            layers[role] = link(relative, layer["path"], layer["sha256"])
        variant = "saved" if "saved" in ref["snapshots"] else "draft"
        snapshot = ref["snapshots"][variant]
        annotation = next(a for a in record["annotations"] if a["snapshot_variant"] == variant)
        snapshots = {}
        for item in record["annotations"]:
            relative = item["provenance"]
            digest = hashes.file(data._path(reference, relative), "reference/" + relative)
            snapshots[item["snapshot_variant"]] = {"provenance_path": relative,
                "provenance_sha256": digest, "labels_path": item["path"],
                "labels_sha256": item["sha256"]}
        accepted_observed = {v: "accepted" in s for v, s in ref["snapshots"].items()}
        provenance = {
            "snapshot": variant, "original_provenance": snapshots[variant],
            "snapshots": snapshots, "source_root": str(reference),
            "source_tile": {"path": tile_path, "sha256": tile_sha},
            "verification": {"basis": "user_requested" if verified else "not_requested",
                "reason": "Explicit importer verified=True user request" if verified else "verified=False",
                "independent_certification": False},
            "confidence_observed": "confidence" in snapshot,
            "confidence_status": "preserved" if "confidence" in snapshot else "unknown",
            "accepted_observed": accepted_observed,
            "known_accepted_cells_union": int(ref["accepted"].sum()),
        }
        annotations = {
            "painting": array(prefix + "_painting.npy", snapshot["painting"]),
            "labels": link(prefix + "_labels.tif", annotation["path"], annotation["sha256"]),
            "verified": verified, "label_status": status, "provenance": provenance,
        }
        if any(accepted_observed.values()) or ref["accepted"].any():
            annotations["accepted"] = array(prefix + "_accepted.npy", ref["accepted"].astype(bool))
        if "confidence" in snapshot:
            annotations["confidence"] = array(prefix + "_confidence.npy", snapshot["confidence"])
        site_info = sites.get(site, {})
        aliases = [data.REGIONS[site], site, site_info.get("map_id"),
                   site_info.get("title"), record.get("legacy_name")]
        records.append({
            "working_dem": layers["dem"], "tile_id": tile_id, "site": site,
            "group": data.GROUPS[site],
            "geography_aliases": list(dict.fromkeys(a for a in aliases if a)),
            "section": target.name, "collection": "reference", "restricted": True,
            "size": list(grid.shape), "transform": list(grid.transform)[:6],
            "crs": grid.crs.to_wkt(), "grid": grid.metadata(), "label_status": status,
            "annotations": annotations, "layers": layers,
            "feature_sources": [Path(os.path.relpath(p, target)).as_posix() for p in sources[tile_id]],
            "source_refs": {"root": str(reference), "tile": tile_path, "sha256": tile_sha,
                            "restricted": True, "download_allowed": False},
        })
        map_id = record.get("map_id") or site_info.get("map_id")
        if map_id:
            records[-1]["map_id"] = map_id
        if site_info.get("title"):
            records[-1]["title"] = site_info["title"]
    small_files["working_dems.json"] = _json_bytes(records)
    small_files["README.md"] = README.format(section=target.name).encode("utf-8")
    for name in ("working_dems.json", "README.md"):
        materialized[name] = {"sha256": hashlib.sha256(small_files[name]).hexdigest(),
                              "storage": "generated_metadata"}
    provenance = {
        "format": "selenograph-processed-reference-v1", "restricted": True,
        "section": target.name, "site_id": records[0]["site"],
        "reference_artifacts_verified": count, "verified": verified,
        "label_status": status, "geographic_groups": data.GROUPS,
        "public_regions": data.REGIONS, "source_root": str(reference),
        "source_hashes": hashes.records, "materialized_inputs": materialized,
        "tile_count": len(records), "download_allowed": False,
    }
    small_files["dataset.json"] = _json_bytes(provenance)
    return _Plan(target, records, provenance, links, small_files, dict(hashes.cache))


def _validate_existing(plan):
    target = plan.target
    if target.is_symlink():
        raise ValueError(f"Output must not be a symlink: {target}")
    if not target.exists() or not any(target.iterdir()):
        return False
    expected = set(plan.links) | set(plan.small_files)
    actual = set()
    for path in target.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Existing output contains a symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(target).as_posix())
    if actual != expected:
        raise ValueError("Refusing nonempty output: published file inventory differs")
    for name, content in plan.small_files.items():
        if (target / name).read_bytes() != content:
            raise ValueError(f"Refusing nonempty output: manifest/materialized content differs: {name}")
    hashes = data._Hashes()
    for name in plan.links:
        hashes.file(target / name, name, plan.provenance["materialized_inputs"][name]["sha256"])
    return True


def _report(plans, existing, pending_status):
    sections = {}
    for plan, exists in zip(plans, existing):
        sections[plan.target.name] = {
            "status": "already_published" if exists else pending_status,
            "output_root": str(plan.target), "tile_count": len(plan.records),
            "hardlink_count": len(plan.links), "records": plan.records,
            "materialized_inputs": plan.provenance["materialized_inputs"],
        }
    return {"status": "already_published" if all(existing) else pending_status,
            "processed_root": str(plans[0].target.parent), "sections": sections,
            "tile_count": sum(len(plan.records) for plan in plans),
            "hardlink_count": sum(len(plan.links) for plan in plans),
            "records": [record for plan in plans for record in plan.records],
            "reference_artifacts_verified": plans[0].provenance["reference_artifacts_verified"]}


def inventory(processed_root=None, reference_root=None, verified=True, *, site=None):
    """Return combined records and per-section destinations/hashes; write nothing."""
    with rasterio.Env(GDAL_PAM_ENABLED="NO", PROJ_NETWORK="OFF"):
        plans = _plans(processed_root, reference_root, verified, site)
        existing = [_validate_existing(plan) for plan in plans]
    return _report(plans, existing, "dry_run")


def run(processed_root=None, reference_root=None, verified=True, *, site=None):
    """Publish each source site directly into processed_root/<site_id>(reference).

    All selected plans and existing destinations are validated before any writes.
    All missing sections are staged before the first rename; each site's publication
    is atomic, not the batch. Re-running after an interruption validates completed
    sites and resumes the rest without replacing a nonempty destination.

    Returns combined records/counts and a sections dict keyed by folder name, with
    each section's status, absolute output_root and materialized input hashes.
    site optionally selects one source site ID; omitted means all preserved sites.
    """
    with rasterio.Env(GDAL_PAM_ENABLED="NO", PROJ_NETWORK="OFF"):
        plans = _plans(processed_root, reference_root, verified, site)
        existing = [_validate_existing(plan) for plan in plans]
        stages = []
        try:
            for plan, exists in zip(plans, existing):
                if exists:
                    continue
                # Hidden sibling staging stays on the same filesystem as the target.
                stage = Path(tempfile.mkdtemp(prefix=".reference-stage-", dir=plan.target.parent))
                stages.append((plan, stage))
                for relative, source in plan.links.items():
                    destination = stage / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(source, destination)
                    except OSError as exc:
                        raise RuntimeError(f"Hard link unavailable for {source}: {exc}. "
                                           "Use the same filesystem with hard-link support; "
                                           "no large-copy fallback is permitted.") from exc
                for relative, content in plan.small_files.items():
                    destination = stage / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
            # Recheck every input and pending target before publishing any section.
            hashes = data._Hashes()
            for plan, _ in stages:
                for path, expected in plan.input_hashes.items():
                    hashes.file(path, str(path), expected)
                if plan.target.is_symlink():
                    raise ValueError("Output became a symlink during publication")
                if plan.target.exists() and any(plan.target.iterdir()):
                    raise ValueError("Output became nonempty during publication; refusing replacement")
            for plan, stage in stages:
                # POSIX rename replaces an empty directory, never a populated one.
                stage.rename(plan.target)
        finally:
            for _, stage in stages:
                if stage.exists():
                    shutil.rmtree(stage)
    return _report(plans, existing, "published")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-root", type=Path, help="Existing processed tree (default: paths.DEM_DIR)")
    parser.add_argument("--reference-root", type=Path, help="Preserved maps (default: paths.PROFESSOR_MAPS_DIR)")
    parser.add_argument("--site", help="Source site ID to publish (default: all preserved sites)")
    parser.add_argument("--verified", action=argparse.BooleanOptionalAction, default=True,
                        help="Explicit user-marked verification, NOT independent certification (default: true)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print inventory; write nothing")
    args = parser.parse_args(argv)
    operation = inventory if args.dry_run else run
    try:
        result = operation(args.processed_root, args.reference_root, verified=args.verified, site=args.site)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Reference import refused: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
