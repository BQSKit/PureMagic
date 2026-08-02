# Repository instructions

## Commit messages

**Never mention Claude, Anthropic, or any AI assistant in a commit message.**
This includes:

- `Co-Authored-By:` trailers naming an assistant or `noreply@anthropic.com`
- "Generated with", "written by", or any similar attribution
- Mentioning an assistant in the subject or body, even incidentally

Write commit messages as the repository's own authors, describing the change
and the reasoning behind it. If a change touches assistant-related
configuration, name the files and settings involved without naming the tool.

Match the existing commit style: an imperative subject line, then a body that
explains *why* the change is being made, with `-` bullets for multi-part
changes and concrete measurements where they exist.
