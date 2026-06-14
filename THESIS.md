# Local thesis rendering

The ITMO PhD thesis template is tracked as a submodule:

```bash
git submodule update --init itmo-phd-thesis-template-en
```

Edit text in `itmo-phd-thesis-template-en/Dissertation/*.tex`, bibliography in
`itmo-phd-thesis-template-en/biblio/*.bib`, then render:

```bash
make thesis-pdf
```

The rendered PDF is written to `outputs/thesis/thesis-itmo.pdf`.

The local render uses `tectonic` and applies small compatibility patches in a
temporary build copy, leaving the upstream submodule files untouched.

By default the target renders a draft PDF without running real `biber`, because
the current Tectonic bundle and Homebrew `biber` versions are incompatible. For
a full bibliography run, install a matched TeX Live/biber pair and run:

```bash
THESIS_FULL_BIB=1 make thesis-pdf
```
