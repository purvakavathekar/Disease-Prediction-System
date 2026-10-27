# UI redesign notes

Only the presentation layer changed. All Python files, datasets, trained models, reports and
`disease_history.db` are byte-identical to the original ZIP.

## Design system
- `static/stylesheets/auth.css`    sign-in / registration (full-screen muted-blue inquiry form)
- `static/stylesheets/archive.css` signed-in case-file pages; scoped to `body.archive`, loaded after
  Bootstrap and `style.css`, so existing scripts keep working and are restyled through the same class names
- `static/stylesheets/fonts.css` + `static/fonts/` self-hosted Gloock and IBM Plex Sans (SIL OFL 1.1)
- `static/js/auth-form.js`         inline validation messages (same rules as the HTML attributes)
- `templates/_archive_head.html`, `_archive_top.html`, `_archive_bottom.html`  shared shell (folder tabs, flashes)

`static/stylesheets/style.css` is untouched and still loaded.

## Design tokens (archive.css, section 1)
paper #F8F2E4, desk #EADFC6, ink #141414, blue #1F4A85, urgent #9B1C24, caution #7A4E00, ok #235C35.
Display type: Gloock. Interface and body: IBM Plex Sans.
