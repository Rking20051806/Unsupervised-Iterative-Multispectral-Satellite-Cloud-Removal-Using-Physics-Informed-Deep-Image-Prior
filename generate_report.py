import os

def escape_latex(text):
    # Escape special LaTeX characters
    chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}'
    }
    return ''.join(chars.get(c, c) for c in text)

def generate_tex():
    root_dir = r"c:\Users\01soj\Downloads\Pinn+dipp"
    ignore_dirs = {".venv", "__pycache__", ".git", "graphify-out", ".pytest_cache"}
    
    tex_content = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{geometry}",
        r"\usepackage{hyperref}",
        r"\geometry{a4paper, margin=1in}",
        r"\title{Project Files Report}",
        r"\author{Generated automatically}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage"
    ]

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Modify dirnames in-place to skip ignored directories
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == ".":
            section_name = "Root Directory"
        else:
            section_name = escape_latex(rel_path)
            
        tex_content.append(r"\section{" + section_name + "}")
        
        if not filenames:
            tex_content.append("No files in this directory.\\\\")
        else:
            tex_content.append(r"\begin{itemize}")
            for f in sorted(filenames):
                tex_content.append(r"    \item " + escape_latex(f))
            tex_content.append(r"\end{itemize}")

    tex_content.append(r"\end{document}")
    
    with open(os.path.join(root_dir, "report.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex_content))

if __name__ == "__main__":
    generate_tex()
