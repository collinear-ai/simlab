# pdf-xlsx-reporting

Use this skill when a task asks you to summarize or extract data from PDF and XLSX files that
already exist in the workspace.

## Preferred workflow

1. Use `run_bash` with `pdftotext <pdf_path> -` to extract readable text from PDFs.
2. Use `run_bash` with `xlsx2csv <xlsx_path>` to convert spreadsheets to CSV.
3. Use `read_file` to inspect any saved intermediate files when needed.
4. Write the final report as markdown with `write_file`.

## Do not do this first

- Do not write custom Python parsers for PDF or XLSX unless the CLI tools fail.
- Do not guess spreadsheet values without converting the file first.

## Typical commands

```bash
pdftotext fixtures/quarterly_metrics.pdf -
```

```bash
xlsx2csv fixtures/headcount.xlsx > /tmp/headcount.csv
```
