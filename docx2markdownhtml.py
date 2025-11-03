# === FULLY FIXED VERSION - Multi-Level Comments with Tree Structure ===

# 1. Install packages
!pip install -q 'markitdown[all]' python-docx lxml markdown mammoth

# 2. Import libraries
from google.colab import files
from markitdown import MarkItDown
from docx import Document
from lxml import etree
import zipfile
from datetime import datetime
from IPython.display import Markdown, display, HTML
import markdown
import mammoth
from collections import defaultdict

# 3. XML Namespaces
ooXMLns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
w15XMLns = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'w15': 'http://schemas.microsoft.com/office/word/2012/wordml'
}

# 4. Enhanced helper functions with threading support
def get_document_comments_with_threading(docx_filename):
    """Extract all comments with metadata and thread relationships"""
    comments_dict = {}
    author_dict = {}
    date_dict = {}
    para_id_dict = {}
    thread_dict = {}  # Maps comment ID to parent ID
    
    try:
        docx_zip = zipfile.ZipFile(docx_filename)
        
        # Extract basic comment data from comments.xml
        comments_xml = docx_zip.read('word/comments.xml')
        et = etree.XML(comments_xml)
        comments = et.xpath('//w:comment', namespaces=ooXMLns)
        
        for c in comments:
            comment_id = c.xpath('@w:id', namespaces=ooXMLns)[0]
            comments_dict[comment_id] = c.xpath('string(.)', namespaces=ooXMLns)
            author_dict[comment_id] = c.xpath('@w:author', namespaces=ooXMLns)[0]
            date_dict[comment_id] = c.xpath('@w:date', namespaces=ooXMLns)[0]
            
            # Try to get paraId from w:p element
            para_id = c.xpath('.//w:p/@w15:paraId', namespaces=w15XMLns)
            if para_id:
                para_id_dict[comment_id] = para_id[0]
        
        print(f"✅ Found {len(comments_dict)} comments")
        
        # Extract threading information from commentsExtended.xml (Office 2013+)
        try:
            comments_ext_xml = docx_zip.read('word/commentsExtended.xml')
            et_ext = etree.XML(comments_ext_xml)
            
            # Parse comment extensions with parent relationships
            comment_exts = et_ext.xpath('//w15:commentEx', namespaces=w15XMLns)
            
            # Build para_id to comment_id mapping
            para_to_comment = {}
            for comment_id, para_id in para_id_dict.items():
                para_to_comment[para_id] = comment_id
            
            for c_ext in comment_exts:
                para_id = c_ext.xpath('@w15:paraId', namespaces=w15XMLns)[0]
                para_id_parent = c_ext.xpath('@w15:paraIdParent', namespaces=w15XMLns)
                
                if para_id_parent:
                    # This comment is a reply
                    comment_id = para_to_comment.get(para_id)
                    parent_comment_id = para_to_comment.get(para_id_parent[0])
                    
                    if comment_id and parent_comment_id:
                        thread_dict[comment_id] = parent_comment_id
            
            print(f"✅ Found {len(thread_dict)} threaded replies")
            
        except KeyError:
            print("ℹ️ No commentsExtended.xml found (using legacy comment format)")
    
    except Exception as e:
        print(f"ℹ️ Error extracting comments: {e}")
    
    return comments_dict, author_dict, date_dict, thread_dict

def get_commented_text(docx_filename):
    """Extract text that has comments attached"""
    comments_of_dict = {}
    try:
        docx_zip = zipfile.ZipFile(docx_filename)
        document_xml = docx_zip.read('word/document.xml')
        et = etree.XML(document_xml)
        comment_ranges = et.xpath('//w:commentRangeStart', namespaces=ooXMLns)
        
        for c in comment_ranges:
            comment_id = c.xpath('@w:id', namespaces=ooXMLns)[0]
            parts = et.xpath(
                f"//w:r[preceding-sibling::w:commentRangeStart[@w:id={comment_id}] "
                f"and following-sibling::w:commentRangeEnd[@w:id={comment_id}]]",
                namespaces=ooXMLns
            )
            comments_of_dict[comment_id] = ''.join(
                part.xpath('string(.)', namespaces=ooXMLns) for part in parts
            )
    except: pass
    return comments_of_dict

def build_comment_tree(comments_dict, thread_dict):
    """Build hierarchical tree structure from comment relationships"""
    tree = defaultdict(list)  # parent_id -> [child_ids]
    root_comments = []
    
    # Identify root comments (those without parents)
    for comment_id in comments_dict.keys():
        if comment_id in thread_dict:
            parent_id = thread_dict[comment_id]
            tree[parent_id].append(comment_id)
        else:
            root_comments.append(comment_id)
    
    return tree, root_comments

# 5. HTML Template with threaded comment styling
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Converted Document</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
            background: #f8f9fa;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }
        h3 { color: #7f8c8d; margin-top: 20px; }
        h4 { color: #95a5a6; margin-top: 15px; margin-left: 20px; }
        h5 { color: #a0a0a0; margin-top: 12px; margin-left: 40px; }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        blockquote {
            border-left: 4px solid #3498db;
            margin: 20px 0;
            padding: 10px 20px;
            background: #ecf0f1;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) { background-color: #f9f9f9; }
        sup { color: #e74c3c; font-weight: bold; }
        hr { border: none; border-top: 2px solid #ecf0f1; margin: 30px 0; }
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""

# 6. Enhanced Parser with Threading Support
class DocxMarkdownParser:
    def __init__(self):
        self.md_converter = MarkItDown(style_map="comment-reference => sup")
        self.docx_filename = None
        print("✅ DocxMarkdownParser initialized with threading support")
    
    def parse(self, docx_path, output_md="output.md", output_html="output.html"):
        """Convert DOCX to both Markdown and HTML with threaded comments"""
        self.docx_filename = docx_path
        print(f"\n🔄 Processing: {docx_path}")
        
        # Extract comments with threading information
        comments, authors, dates, thread_dict = get_document_comments_with_threading(docx_path)
        commented_text = get_commented_text(docx_path)
        
        # Build comment tree structure
        comment_tree, root_comments = build_comment_tree(comments, thread_dict)
        
        # Convert main content to Markdown
        print("🔄 Converting to Markdown...")
        result = self.md_converter.convert(docx_path)
        
        # Enhance with threaded comment details
        enhanced_md = self._add_threaded_comments(
            result.text_content,
            comments,
            authors,
            dates,
            commented_text,
            thread_dict,
            comment_tree,
            root_comments
        )
        
        # Save Markdown
        with open(output_md, "w", encoding='utf-8') as f:
            f.write(enhanced_md)
        print(f"✅ Markdown saved: {output_md}")
        
        # Convert to HTML
        print("🔄 Converting to HTML...")
        html_content = self._convert_to_html(enhanced_md)
        
        with open(output_html, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ HTML saved: {output_html}")
        
        return enhanced_md, html_content
    
    def _add_threaded_comments(self, markdown_content, comments, authors, dates, 
                                commented_text, thread_dict, comment_tree, root_comments):
        """Add hierarchical comment section to markdown"""
        
        if not comments:
            return markdown_content
        
        markdown_content += "\n\n---\n\n## 💬 Document Comments (Threaded)\n\n"
        
        def format_comment(comment_id, level=0):
            """Recursively format comment and its replies"""
            date_str = dates.get(comment_id, "")
            
            if date_str:
                try:
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    date_str = date_obj.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            # Build the output with proper formatting
            output = ""
            
            # Format comment header based on level
            if level == 0:
                # Root comment - use h3
                output += f"### Comment [{comment_id}]\n\n"
                output += f"**Author**: {authors.get(comment_id, 'Unknown')}\n\n"
                
                if date_str:
                    output += f"**Date**: {date_str}\n\n"
                
                # Show referenced text for root comments
                if comment_id in commented_text and commented_text[comment_id].strip():
                    output += f"**Referenced Text**: \"{commented_text[comment_id]}\"\n\n"
                
                output += f"**Comment**: {comments[comment_id]}\n\n"
                
            else:
                # Reply comment - use bullet points with indentation
                indent = "  " * (level - 1)
                parent_id = thread_dict.get(comment_id, "Unknown")
                
                output += f"{indent}- **Reply to [{parent_id}]** (Comment ID: {comment_id})\n"
                output += f"{indent}  - Author: {authors.get(comment_id, 'Unknown')}\n"
                
                if date_str:
                    output += f"{indent}  - Date: {date_str}\n"
                
                output += f"{indent}  - Comment: {comments[comment_id]}\n\n"
            
            # Recursively add replies
            if comment_id in comment_tree:
                for reply_id in sorted(comment_tree[comment_id]):
                    output += format_comment(reply_id, level + 1)
            
            # Add separator after root comments
            if level == 0:
                output += "---\n\n"
            
            return output
        
        # Format root comments and their threads
        for comment_id in sorted(root_comments):
            markdown_content += format_comment(comment_id)
        
        return markdown_content
    
    def _convert_to_html(self, markdown_text):
        """Convert Markdown to styled HTML"""
        html_body = markdown.markdown(
            markdown_text,
            extensions=['extra', 'codehilite', 'tables', 'fenced_code']
        )
        return HTML_TEMPLATE.format(content=html_body)

# 7. Upload file
print("=" * 60)
print("📁 STEP 1: Upload your DOCX file")
print("=" * 60)
uploaded = files.upload()
docx_filename = list(uploaded.keys())[0]
print(f"✅ Uploaded: {docx_filename}\n")

# 8. Convert with threading support
print("=" * 60)
print("📄 STEP 2: Converting document with threading support")
print("=" * 60)
parser = DocxMarkdownParser()
markdown_output, html_output = parser.parse(
    docx_filename,
    output_md="output.md",
    output_html="output.html"
)

# 9. Display statistics
print("\n" + "=" * 60)
print("📊 CONVERSION STATISTICS")
print("=" * 60)
import os
print(f"  - output.md: {os.path.getsize('output.md'):,} bytes")
print(f"  - output.html: {os.path.getsize('output.html'):,} bytes")

# 10. Preview
print("\n" + "=" * 60)
print("👀 MARKDOWN PREVIEW")
print("=" * 60)
print(markdown_output[:1000])
print("..." if len(markdown_output) > 1000 else "")

# 11. Download files
print("\n" + "=" * 60)
print("⬇️ STEP 3: Downloading files")
print("=" * 60)
files.download("output.md")
files.download("output.html")

print("\n✅ CONVERSION COMPLETE!")
print("   • Markdown includes threaded comment tree structure")
print("   • HTML displays hierarchical comment relationships")
