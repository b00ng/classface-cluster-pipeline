# === COLAB ALL-IN-ONE: DOCX → Markdown + HTML with THREADED COMMENTS ===
# Installs, extraction helpers, threaded mapping, converter class, upload UI, preview, and downloads.

# 1) Install packages
!pip install -q 'markitdown[all]' python-docx lxml markdown mammoth

# 2) Imports
from google.colab import files
from markitdown import MarkItDown
from lxml import etree
import zipfile
from datetime import datetime
from collections import defaultdict
import markdown
import mammoth
import os

# 3) XML namespaces (Word 2006 main + Word 2012 w15)
NS_W = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
NS_W15 = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'w15': 'http://schemas.microsoft.com/office/word/2012/wordml'
}

# 4) Robust threaded comments extraction based on comments.xml + commentsExtended.xml
def extract_comments_with_first_paraid(docx_path):
    """
    Parse word/comments.xml and capture:
      - commentId -> {text, author, date, first_paraId}
    Use the FIRST paragraph's w15:paraId as the canonical anchor for each comment.
    """
    comments = {}
    paraid_by_comment = {}
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read('word/comments.xml')
    root = etree.XML(xml)

    for c in root.xpath('//w:comment', namespaces=NS_W):
        cid = c.xpath('@w:id', namespaces=NS_W)[0]
        text = c.xpath('string(.)', namespaces=NS_W)
        author = (c.xpath('@w:author', namespaces=NS_W) or ['Unknown'])[0]
        date = (c.xpath('@w:date', namespaces=NS_W) or [''])[0]

        # First paragraph paraId (canonical for threading)
        first_p = c.xpath('.//w:p[1]', namespaces=NS_W)
        para_id = ''
        if first_p:
            p_vals = first_p[0].xpath('@w15:paraId', namespaces=NS_W15)
            if p_vals:
                para_id = p_vals[0]

        comments[cid] = {
            'text': text,
            'author': author,
            'date': date,
            'first_paraId': para_id
        }
        paraid_by_comment[cid] = para_id

    return comments, paraid_by_comment

def extract_thread_edges(docx_path, paraid_by_comment):
    """
    Read word/commentsExtended.xml and join w15:paraIdParent → parent paraId
    to build child_comment_id → parent_comment_id mapping.
    Falls back to empty edges if commentsExtended.xml is absent.
    """
    comment_by_para = {p: cid for cid, p in paraid_by_comment.items() if p}
    edges = {}
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml = z.read('word/commentsExtended.xml')
    except KeyError:
        # No threading information (legacy comments)
        return edges

    root = etree.XML(xml)
    for cx in root.xpath('//w15:commentEx', namespaces=NS_W15):
        para = (cx.xpath('@w15:paraId', namespaces=NS_W15) or [''])[0]
        parent_para = (cx.xpath('@w15:paraIdParent', namespaces=NS_W15) or [''])[0]
        if para and parent_para:
            child_cid = comment_by_para.get(para, '')
            parent_cid = comment_by_para.get(parent_para, '')
            if child_cid and parent_cid and child_cid != parent_cid:
                edges[child_cid] = parent_cid

    return edges

def build_thread_tree(comments, edges):
    """
    Build a hierarchical tree:
      - children_by_parent: parent_id -> [child_ids]
      - roots: list of comment_ids with no parent
    Sort roots and children chronologically by comment date (ISO) for stable output.
    """
    children = defaultdict(list)
    has_parent = set(edges.keys())

    for child, parent in edges.items():
        if child in comments and parent in comments:
            children[parent].append(child)

    roots = [cid for cid in comments.keys() if cid not in has_parent]

    def sort_key(cid):
        d = comments[cid].get('date', '')
        try:
            return (datetime.fromisoformat(d.replace('Z', '+00:00')), cid)
        except Exception:
            return (datetime.min, cid)

    roots.sort(key=sort_key)
    for parent in children:
        children[parent].sort(key=sort_key)

    return children, roots

# 5) Extract referenced text for each commentId from document.xml
def extract_commented_text(docx_path):
    """
    Map commentId -> referenced text (text between w:commentRangeStart/@w:id and w:commentRangeEnd/@w:id).
    """
    refs = {}
    with zipfile.ZipFile(docx_path) as z:
        doc_xml = z.read('word/document.xml')
    et = etree.XML(doc_xml)

    starts = et.xpath('//w:commentRangeStart', namespaces=NS_W)
    for s in starts:
        cid = s.xpath('@w:id', namespaces=NS_W)[0]
        # Select runs between start and end with matching id
        parts = et.xpath(
            f"//w:r[preceding-sibling::w:commentRangeStart[@w:id={cid}] "
            f"and following-sibling::w:commentRangeEnd[@w:id={cid}]]",
            namespaces=NS_W
        )
        text = ''.join(p.xpath('string(.)', namespaces=NS_W) for p in parts)
        refs[cid] = text
    return refs

# 6) HTML template for pretty output
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Converted Document</title>
<style>
 body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; line-height:1.6; max-width:900px; margin:40px auto; padding:0 20px; color:#333; background:#f8f9fa; }
 .container { background:#fff; padding:40px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,0.1); }
 h1 { color:#2c3e50; border-bottom:3px solid #3498db; padding-bottom:10px; }
 h2 { color:#34495e; margin-top:30px; border-bottom:2px solid #ecf0f1; padding-bottom:8px; }
 h3 { color:#7f8c8d; margin-top:20px; }
 code { background:#f4f4f4; padding:2px 6px; border-radius:3px; font-family:'Courier New',monospace; }
 pre { background:#2d2d2d; color:#f8f8f2; padding:15px; border-radius:5px; overflow-x:auto; }
 table { border-collapse:collapse; width:100%; margin:20px 0; }
 th,td { border:1px solid #ddd; padding:12px; text-align:left; }
 th { background:#3498db; color:#fff; }
 tr:nth-child(even){ background:#f9f9f9; }
 .comment-root { background:#fff3cd; border-left:4px solid #ffc107; padding:15px; margin:15px 0; border-radius:4px; }
 .comment-reply { margin-left:30px; background:#e3f2fd; border-left:4px solid #2196f3; padding:12px; margin-top:10px; border-radius:4px; }
 .comment-reply-2 { margin-left:60px; background:#f3e5f5; border-left:4px solid #9c27b0; padding:10px; margin-top:8px; border-radius:4px; }
 .meta { color:#95a5a6; font-size:0.9em; }
 hr { border:none; border-top:2px solid #ecf0f1; margin:30px 0; }
 sup { color:#e74c3c; font-weight:bold; }
</style>
</head>
<body><div class="container">{content}</div></body></html>
"""

# 7) Markdown formatting for threaded comments
def add_threaded_comments_markdown(base_md, comments, children, roots, referenced_text):
    md = base_md + "\n\n---\n\n## 💬 Document Comments (Threaded)\n\n"
    def fmt_date(d):
        try:
            return datetime.fromisoformat(d.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
        except Exception:
            return d

    def emit(cid, level=0):
        c = comments[cid]
        if level == 0:
            block = f"### Comment [{cid}]\n\n"
            block += f"**Author**: {c['author']}\n\n"
            dt = fmt_date(c.get('date', ''))
            if dt:
                block += f"**Date**: {dt}\n\n"
            # Show referenced text only for root
            if cid in referenced_text and referenced_text[cid].strip():
                block += f"**Referenced Text**: \"{referenced_text[cid]}\"\n\n"
            block += f"**Comment**: {c['text']}\n\n"
        else:
            indent = "  " * (level - 1)
            block = f"{indent}- **Reply** (ID: {cid})\n"
            block += f"{indent}  - Author: {c['author']}\n"
            dt = fmt_date(c.get('date', ''))
            if dt:
                block += f"{indent}  - Date: {dt}\n"
            block += f"{indent}  - Comment: {c['text']}\n\n"

        for child in children.get(cid, []):
            block += emit(child, level + 1)

        if level == 0:
            block += "---\n\n"
        return block

    for r in roots:
        md += emit(r, 0)
    return md

# 8) HTML conversion using python-markdown
def markdown_to_html(styled_md):
    html_body = markdown.markdown(
        styled_md,
        extensions=['extra', 'codehilite', 'tables', 'fenced_code']
    )
    return HTML_TEMPLATE.format(content=html_body)

# 9) Optional: Direct DOCX → HTML (for fidelity comparison)
def docx_to_html_direct(docx_path, out_html="output_direct.html"):
    with open(docx_path, "rb") as f:
        res = mammoth.convert_to_html(f)
    styled = HTML_TEMPLATE.format(content=res.value)
    with open(out_html, "w", encoding="utf-8") as g:
        g.write(styled)
    return styled

# 10) Converter class
class DocxMarkdownParser:
    def __init__(self):
        # Show comment references in-text as superscripts
        self.md_converter = MarkItDown(style_map="comment-reference => sup")

    def parse(self, docx_path, out_md="output.md", out_html="output.html"):
        # Extract core comments and threading info
        comments, paraid_by_comment = extract_comments_with_first_paraid(docx_path)
        edges = extract_thread_edges(docx_path, paraid_by_comment)
        children_by_parent, roots = build_thread_tree(comments, edges)
        referenced = extract_commented_text(docx_path)

        # Convert main content to Markdown via MarkItDown
        result = self.md_converter.convert(docx_path)
        base_md = result.text_content

        # Append threaded comments as Markdown
        enhanced_md = add_threaded_comments_markdown(
            base_md=base_md,
            comments=comments,
            children=children_by_parent,
            roots=roots,
            referenced_text=referenced
        )

        # Save Markdown
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(enhanced_md)

        # Convert to styled HTML and save
        html = markdown_to_html(enhanced_md)
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)

        return enhanced_md, html

# 11) Upload, convert, preview, download
print("="*60)
print("📁 Upload a DOCX file with comments")
uploaded = files.upload()
docx_filename = list(uploaded.keys())[0]
print(f"✅ Uploaded: {docx_filename}")

parser = DocxMarkdownParser()
md_text, html_text = parser.parse(docx_filename, "output.md", "output.html")

# Optional: also create direct HTML for comparison
direct_html = docx_to_html_direct(docx_filename, "output_direct.html")

print("\n📊 Outputs:")
print(f"  • output.md: {os.path.getsize('output.md'):,} bytes")
print(f"  • output.html: {os.path.getsize('output.html'):,} bytes")
print(f"  • output_direct.html: {os.path.getsize('output_direct.html'):,} bytes")

# Preview beginning of Markdown
print("\n📝 Markdown preview (first 800 chars):\n")
print(md_text[:800])
print("..." if len(md_text) > 800 else "")

# Download all outputs
print("\n⬇️ Downloading files...")
files.download("output.md")
files.download("output.html")
files.download("output_direct.html")
print("✅ Done!")
