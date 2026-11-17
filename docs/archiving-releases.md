# Archiving software releases

The paper has a [Springer DOI](https://doi.org/10.1007/978-3-032-33010-9_10).
A software DOI identifies an archived code release, a separate research output.
The repository's [`CITATION.cff`](../CITATION.cff) describes the software and
names the conference paper as the preferred citation.

## One-time Zenodo setup

1. Sign in to [Zenodo](https://zenodo.org/) and connect the GitHub account that
   owns this repository. Connect ORCID as well if desired.
2. Open the GitHub integration page from the Zenodo profile menu and select
   **Sync now**.
3. Enable `dariocazzani/intelligence-entropy` and confirm that it is connected.
4. Merge the validated citation metadata before making a new software release.

See Zenodo's [account-linking guide](https://help.zenodo.org/docs/profile/linking-accounts/)
and [repository setup guide](https://help.zenodo.org/docs/github/enable-repository/).
Enabling the integration does not retroactively archive the existing
`agi-2026-camera-ready` release; preserve that tag as the submitted snapshot.

## For each new software release

1. Choose the version and update the package version and `CITATION.cff`
   `version`/`date-released` fields together. Run tests and validate the CFF.
2. Create a new GitHub tag and release for the tested commit.
3. Wait for Zenodo to ingest it. Inspect the record's author, title, license,
   version, and archived files before sharing its DOI.
4. Verify that the software record identifies the Springer DOI as its related
   publication. Add that relationship in Zenodo if it was not imported.
5. Add the software DOI and archive link to the repository. For reproducibility,
   cite the version DOI that identifies the actual code used; the concept DOI
   represents the software across versions.

Do not put the paper DOI in the software's top-level CFF `doi` field. Keep it
under `preferred-citation`; a software DOI belongs at the top level only after
the software archive exists. The metadata deliberately does not claim an
uncreated software DOI.

Zenodo can read `CITATION.cff` directly. Add `.zenodo.json` only if additional
Zenodo-specific metadata is necessary, since it takes precedence over the CFF.
See [Zenodo's release guide](https://help.zenodo.org/docs/github/archive-software/github-upload/)
and [metadata precedence](https://help.zenodo.org/docs/github/describe-software/zenodo-json/).
