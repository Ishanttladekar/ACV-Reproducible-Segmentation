import markdown

with open("d:/ACV/23115043.md", "r", encoding="utf-8") as f:
    text = f.read()

html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])

template = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ACV Assignment B - 23115043</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #24292e;
    max-width: 960px;
    margin: 40px auto;
    padding: 0 30px;
}}
h1, h2, h3 {{
    border-bottom: 1px solid #eaecef;
    padding-bottom: 0.3em;
    color: #1b1f23;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1.5em 0;
}}
th, td {{
    border: 1px solid #dfe2e5;
    padding: 8px 14px;
    text-align: left;
}}
th {{
    background-color: #f6f8fa;
    font-weight: 600;
}}
tr:nth-child(2n) {{
    background-color: #f8f9fa;
}}
pre {{
    background-color: #f6f8fa;
    border-radius: 4px;
    padding: 16px;
    overflow: auto;
    font-family: Consolas, monospace;
    font-size: 85%;
    line-height: 1.45;
}}
code {{
    background-color: rgba(27,31,35,0.05);
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-family: Consolas, monospace;
    font-size: 85%;
}}
@media print {{
    body {{ max-width: 100%; margin: 0; padding: 12mm; }}
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open("d:/ACV/23115043.html", "w", encoding="utf-8") as f:
    f.write(template)

print("Report HTML successfully generated at d:/ACV/23115043.html")
