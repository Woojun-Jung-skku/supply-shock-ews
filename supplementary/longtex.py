"""Read the table blocks of the extended version (../extended_results): label -> caption, size command, tabular, note.
The environment variable SUPP_LONG sets another folder."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
SUPP = os.environ.get("SUPP_LONG", os.path.join(os.path.dirname(HERE), "extended_results"))
FILES = ["supplementary.tex", "supplementary_extra.tex", "supplementary_part2.tex", "supplementary_backtest.tex", "supplementary_note17_rare_arm.tex"]
def _balanced(s, i):
    """s[i] == '{' -> index after the matching '}'"""
    d = 0
    for j in range(i, len(s)):
        if s[j] == "{" and s[j - 1] != "\\": d += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            d -= 1
            if d == 0: return j + 1
    raise ValueError("unbalanced")
def tables():
    out = {}
    for fn in FILES:
        s = open(os.path.join(SUPP, fn), encoding="utf-8").read()
        for m in re.finditer(r"\\captionof\{table\}\{Table S([0-9.]+?)\. ", s):
            a = s.index("{", m.start() + len("\\captionof{table}")); b = _balanced(s, a)
            cap = s[m.end():b - 1]
            t0 = s.index("\\begin{tabular}", b); t1 = s.index("\\end{tabular}", t0) + len("\\end{tabular}")
            pre = s[b:t0]; size = re.search(r"\\(footnotesize|scriptsize|tiny|small)", pre)
            rest = s[t1:t1 + 2500]; rest = rest.lstrip("}").lstrip()
            stop = re.search(r"\\end\{minipage\}|\\vspace|\\clearpage|\\noindent\\begin\{minipage\}|\\captionof|\\textbf\{|\\end\{landscape\}|\n\s*\n", rest)
            note = rest[:stop.start()].strip() if stop else ""
            out[m.group(1)] = dict(caption=re.sub(r"\\label\{[^}]*\}", "", cap).strip(), size=size.group(1) if size else "footnotesize",
                                   tabular=s[t0:t1], note=note, file=fn, landscape="\\begin{landscape}" in s[max(0, m.start() - 60):m.start()])
    return out
if __name__ == "__main__":
    T = tables(); import sys
    for k in sys.argv[1:]:
        t = T[k]; hdr = [l for l in t["tabular"].split("\n") if "&" in l][0]
        print(f"=== S{k} [{t['size']}{', landscape' if t['landscape'] else ''}] rows={t['tabular'].count(chr(92)*2)}"); print("CAP:", t["caption"][:900]); print("HDR:", hdr[:260]); print("NOTE:", t["note"][:300]); print()
