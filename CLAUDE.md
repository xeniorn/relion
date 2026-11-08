# xeniorn/relion

Our fork of RELION (cryo-EM refinement, upstream 3dem/relion) carrying our patches on top of upstream version
branches; the VBC build lives in `relion_vbc`.

## Conventions

- **Base branches on the head of upstream's version branch** (e.g. `upstream/ver5.1`), not on the release tag (the
  5.1.1 tag lacked GPU allocator fixes already on ver5.1). Every re-fork onto a new upstream commit gets its own new
  branch named after that base: `universal/5.1.1-afdd816` here, `vbc/5.1.1-afdd816` in relion_vbc. Rebase or
  cherry-pick our patches onto the new base in the new branch; old branches stay untouched, no force-pushes.
  Image tags use the same upstream sha7.
- **Full version = `vX.Y.Z-<commit7>`** in image tags, image file names and module names: the version the package
  declares at the pinned commit (setup.py / pyproject / `__version__`) with a leading `v`, plus the upstream commit it
  was built from (e.g. `v5.1.1-afdd816`, `v1.0.18-994945b`). Not the commit alone, not the version without `v`. Ask when
  there is no declared version or no commit rather than inventing one.
