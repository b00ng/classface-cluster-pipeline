# === COLAB ALL-IN-ONE: DOCX → Markdown + HTML with ACCURATE THREADED COMMENTS ===
# Robust mapping using all comment paragraph paraIds, commentsExtended linking, and stable ordering.

# 1) Install packages
!pip install -q 'markitdown[all]' python-docx lxml markdown mammoth

# 2) Imports
from google.colab import files
from markitdown import MarkItDown
from lxml import etree
import zipfile
from datetime import datetime
from collections import defaultdict, OrderedDict
import markdown
import mammoth
import os
import re

# 3) Namespaces
NS_W = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
NS_W15 = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'w15': 'http://schemas.microsoft.com/office/word/2012/wordml'
}

# 4) Utilities
def _norm_hex(s):
    # Normalize paraId hex to uppercase without whitespace for reliable matching
    return s.strip().upper() if isinstance(s, str) else s

def _read_xml_from_zip(z, name):
    try:
        return etree.XML(z.read(name))
    except KeyError:
        return None

def _list_parts(z, pattern):
    # Return sorted part names that match pattern (e.g., r'^word/comments(\d*)\.xml$')
    rx = re.compile(pattern)
    return sorted([n for n in z.namelist() if rx.match(n)])

# 5) Extract comments and ALL paragraph paraIds per comment
def extract_comments_and_paraids(docx_path):
    """
    Return:
      comments: commentId -> {text, author, date}
      paraids_by_comment: commentId -> set(paraId) for ALL paragraphs in the comment
      part_names: list of comment part names (for diagnostics)
    """
    comments = {}
    paraids_by_comment = {}
    with zipfile.ZipFile(docx_path) as z:
        # Support multiple comments parts if present
        part_names = _list_parts(z, r'^word/comments(\d*)\.xml$') or ['word/comments.xml']
        for part in part_names:
            root = _read_xml_from_zip(z, part)
            if root is None:
                continue
            for c in root.xpath('//w:comment', namespaces=NS_W):
                cid = c.xpath('@w:id', namespaces=NS_W)[0]
                text = c.xpath('string(.)', namespaces=NS_W)
                author = (c.xpath('@w:author', namespaces=NS_W) or ['Unknown'])[0]
                date = (c.xpath('@w:date', namespaces=NS_W) or [''])[0]
                # Gather ALL paragraph paraIds within the comment
                para_ids = set()
                for p in c.xpath('.//w:p', namespaces=NS_W):
                    vals = p.xpath('@w15:paraId', namespaces=NS_W15)
                    if vals:
                        para_ids.add(_norm_hex(vals[0]))
                comments[cid] = {'text': text, 'author': author, 'date': date}
                paraids_by_comment[cid] = para_ids
    return comments, paraids_by_comment, part_names

# 6) Extract extended threading and preserve order
def extract_comments_ex_with_order(docx_path):
    """
    Return ordered list of dicts: [{'para': P, 'parent': PP, 'done': 0/1}, ...]
    Reads all commentsExtended*.xml parts in name order, preserving XML order.
    """
    entries = []
    with zipfile.ZipFile(docx_path) as z:
        part_names = _list_parts(z, r'^word/commentsExtended(\d*)\.xml$') or ['word/commentsExtended.xml']
        for part in part_names:
            root = _read_xml_from_zip(z, part)
            if root is None:
                continue
            for cx in root.xpath('//w15:commentEx', namespaces=NS_W15):
                para = (cx.xpath('@w15:paraId', namespaces=NS_W15) or [''])[0]
                ppara = (cx.xpath('@w15:paraIdParent', namespaces=NS_W15) or [''])[0]
                done = (cx.xpath('@w15:done', namespaces=NS_W15) or ['0'])[0]
                entries.append({
                    'para': _norm_hex(para),
                    'parent': _norm_hex(ppara),
                    'done': 1 if str(done) in ('1', 'true', 'True') else 0
                })
    return entries

# 7) Build paraId → commentId index using ALL paraIds
def build_para_to_comment_index(paraids_by_comment):
    idx = {}
    for cid, s in paraids_by_comment.items():
        for pid in s:
            if pid:  # non-empty
                idx[pid] = cid
    return idx

# 8) Build edges (child -> parent) using commentsEx entries and full para index
def build_edges_from_comments_ex(comments_ex_entries, para_to_comment):
    edges = {}          # child_cid -> parent_cid
    node_done = {}      # cid -> done flag (if present)
    order_in_ex = []    # child cids in the order they appear in commentsExtended
    for e in comments_ex_entries:
        child_para = e['para']
        parent_para = e['parent']
        child_cid = para_to_comment.get(child_para)
        parent_cid = para_to_comment.get(parent_para) if parent_para else None
        if child_cid:
            if parent_cid and child_cid != parent_cid:
                edges[child_cid] = parent_cid
            # capture done state
            node_done[child_cid] = e.get('done', 0)
            order_in_ex.append(child_cid)
    # remove duplicates while preserving order
    seen = set()
    order_in_ex_unique = []
    for cid in order_in_ex:
        if cid not in seen:
            order_in_ex_unique.append(cid)
            seen.add(cid)
    return edges, node_done, order_in_ex_unique

# 9) Determine root order by document flow (commentRangeStart order)
def extract_comment_reference_order(docx_path):
    """
    Returns comment IDs in the order they appear in the main document
    using commentRangeStart/@w:id scanning of word/document.xml and headers/footers.
    """
    ids = []
    with zipfile.ZipFile(docx_path) as z:
        doc_roots = []
        main = _read_xml_from_zip(z, 'word/document.xml')
        if main is not None:
            doc_roots.append(main)
        # Also check headers/footers for anchored comments
        for name in _list_parts(z, r'^word/header(\d*)\.xml$') + _list_parts(z, r'^word/footer(\d*)\.xml$'):
            r = _read_xml_from_zip(z, name)
            if r is not None:
                doc_roots.append(r)
        for root in doc_roots:
            for s in root.xpath('//w:commentRangeStart', namespaces=NS_W):
                cid = s.xpath('@w:id', namespaces=NS_W)[0]
                ids.append(cid)
    # de-dup preserving order
    seen, ordered = set(), []
    for cid in ids:
        if cid not in seen:
            ordered.append(cid)
            seen.add(cid)
    return ordered

# 10) Extract referenced text for each commentId (main body only; can be extended)
def extract_commented_text(docx_path):
    refs = {}
    with zipfile.ZipFile(docx_path) as z:
        et = _read_xml_from_zip(z, 'word/document.xml')
        if et is None:
            return refs
        starts = et.xpath('//w:commentRangeStart', namespaces=NS_W)
        for s in starts:
            cid = s.xpath('@w:id', namespaces=NS_W)[0]
            parts = et.xpath(
                f"//w:r[preceding-sibling::w:commentRangeStart[@w:id={cid}] "
                f"and following-sibling::w:commentRangeEnd[@w:id={cid}]]",
                namespaces=NS_W
            )
            text = ''.join(p.xpath('string(.)', namespaces=NS_W) for p in parts)
            refs[cid] = text
    return refs

# 11) Build thread tree with correct ordering (roots by doc order, children by commentsEx order)
def build_thread_tree(comments, edges, roots_order, ex_child_order):
    children = defaultdict(list)
    has_parent = set(edges.keys())
    # Build children lists
    for child, parent in edges.items():
        if child in comments and parent in comments:
            children[parent].append(child)
    # Roots are comments without parent
    roots = [cid for cid in comments.keys() if cid not in has_parent]
    # Order roots by document reference order, then fall back to original key order
    root_order_index = {cid: i for i, cid in enumerate(roots_order)}
    roots.sort(key=lambda cid: (root_order_index.get(cid, 10**9)))
    # Order children by first appearance in commentsExtended
    pos = {cid: i for i, cid in enumerate(ex_child_order)}
    for parent, lst in children.items():
        lst.sort(key=lambda cid: pos.get(cid, 10**9))
    return children, roots

# 12) Markdown formatting
def add_threaded_comments_markdown(base_md, comments, children, roots, referenced_text, done_flags):
    md = base_md + "\n\n---\n\n## 💬 Document Comments (Threaded)\n\n"
    def fmt_date(d):
        try: return datetime.fromisoformat(d.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
        except: return d

    def emit(cid, level=0):
        c = comments[cid]
        is_done = done_flags.get(cid, 0) == 1
        status = " (resolved)" if is_done else ""
        if level == 0:
            block = f"### Comment [{cid}]{status}\n\n"
            block += f"**Author**: {c['author']}\n\n"
            dt = fmt_date(c.get('date', ''))
            if dt: block += f"**Date**: {dt}\n\n"
            if cid in referenced_text and referenced_text[cid].strip():
                block += f"**Referenced Text**: \"{referenced_text[cid]}\"\n\n"
            block += f"**Comment**: {c['text']}\n\n"
        else:
            indent = "  " * (level - 1)
            block = f"{indent}- **Reply** (ID: {cid}){status}\n"
            block += f"{indent}  - Author: {c['author']}\n"
            dt = fmt_date(c.get('date', ''))
            if dt: block += f"{indent}  - Date: {dt}\n"
            block += f"{indent}  - Comment: {c['text']}\n\n"

        for child in children.get(cid, []):
            block += emit(child, level + 1)
        if level == 0: block += "---\n\n"
        return block

    for r in roots:
        md += emit(r, 0)
    return md

# 13) Styled HTML conversion
HTML_TEMPLATE = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Converted Document</title>
<style>
 body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; line-height:1.6; max-width:900px; margin:40px auto; padding:0 20px; color:#333; background:#f8f9fa; }
 .container { background:#fff; padding:40px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,0.1); }
 h1 { color:#2c3e50; border-bottom:3px solid #3498db; padding-bottom:10px; }
 h2 { color:#34495e; margin-top:30px; border-bottom:2px solid #ecf0f1; padding-bottom:8px; }
 h3 { color:#7f8c8d; margin-top:20px; }
 table { border-collapse:collapse; width:100%; margin:20px 0; }
 th,td { border:1px solid #ddd; padding:12px; text-align:left; }
 th { background:#3498db; color:#fff; }
 .resolved { opacity:0.75; }
</style></head><body><div class="container">{content}</div></body></html>"""

def markdown_to_html(styled_md):
    html_body = markdown.markdown(
        styled_md,
        extensions=['extra', 'tables', 'fenced_code']
    )
    # Light touch: add resolved class to headings/bullets containing "(resolved)"
    html_body = re.sub(r'(<h3>[^<]*)(\(resolved\))', r'\1<span class="resolved"> (resolved)</span>', html_body)
    html_body = re.sub(r'(<li>[^<]*)(\(resolved\))', r'\1<span class="resolved"> (resolved)</span>', html_body)
    return HTML_TEMPLATE.format(content=html_body)

# 14) Optional direct HTML
def docx_to_html_direct(docx_path, out_html="output_direct.html"):
    with open(docx_path, "rb") as f:
        res = mammoth.convert_to_html(f)
    styled = HTML_TEMPLATE.format(content=res.value)
    with open(out_html, "w", encoding="utf-8") as g:
        g.write(styled)
    return styled

# 15) Converter class
class DocxMarkdownParser:
    def __init__(self):
        self.md_converter = MarkItDown(style_map="comment-reference => sup")  # show refs inline

    def parse(self, docx_path, out_md="output.md", out_html="output.html"):
        # Comments and paragraph ids
        comments, paraids_by_comment, _ = extract_comments_and_paraids(docx_path)
        # Index paraId -> commentId across ALL paragraphs for robust matching
        para_to_comment = build_para_to_comment_index(paraids_by_comment)
        # Read commentsExtended, keep order
        ex_entries = extract_comments_ex_with_order(docx_path)
        edges, done_flags, ex_child_order = build_edges_from_comments_ex(ex_entries, para_to_comment)
        # Reference order from document for root ordering
        root_order = extract_comment_reference_order(docx_path)
        # Build threaded tree with correct ordering
        children_by_parent, roots = build_thread_tree(comments, edges, root_order, ex_child_order)
        # Referenced text for root comments
        referenced = extract_commented_text(docx_path)

        # Base content via MarkItDown
        result = self.md_converter.convert(docx_path)
        base_md = result.text_content

        # Append threaded comments section
        enhanced_md = add_threaded_comments_markdown(
            base_md, comments, children_by_parent, roots, referenced, done_flags
        )
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(enhanced_md)

        html = markdown_to_html(enhanced_md)
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)

        return enhanced_md, html

# 16) Upload, convert, download
print("="*60)
print("📁 Upload a DOCX file with threaded comments")
uploaded = files.upload()
docx_filename = list(uploaded.keys())[0]
print(f"✅ Uploaded: {docx_filename}")

parser = DocxMarkdownParser()
md_text, html_text = parser.parse(docx_filename, "output.md", "output.html")

# Also produce direct HTML for comparison
direct_html = docx_to_html_direct(docx_filename, "output_direct.html")

print("\n📊 Outputs:")
print(f"  • output.md: {os.path.getsize('output.md'):,} bytes")
print(f"  • output.html: {os.path.getsize('output.html'):,} bytes")
print(f"  • output_direct.html: {os.path.getsize('output_direct.html'):,} bytes")

print("\n📝 Markdown preview (first 900 chars):\n")
print(md_text[:900])
print("..." if len(md_text) > 900 else "")

print("\n⬇️ Downloading files...")
files.download("output.md")
files.download("output.html")
files.download("output_direct.html")
print("✅ Done!")
