# Security Policy

FormCoreset is a local-first tool. It does not run a server, upload images, or
collect telemetry. The main security surface is untrusted input files and
downloaded model weights.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's
[private security advisory](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
for this repository. Include:

- affected version / commit;
- reproduction steps;
- impact (local file access, arbitrary code execution, model download, etc.);
- whether the issue requires processing an untrusted image/model file.

## Scope

- Image files are treated as untrusted data. The tool decodes them with Pillow /
  OpenCV; a decoder vulnerability in those dependencies is in their scope.
- Model weights are downloaded from upstream sources. Users should verify the
  source and license; the project does not bundle weights.
- `dedupe` is non-destructive: it copies outputs and never deletes originals.
- No network access is required after model download.
