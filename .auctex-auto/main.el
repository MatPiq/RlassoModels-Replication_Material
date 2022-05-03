(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("amsart" "11pt" "reqno")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("natbib" "authoryear") ("xy" "all" "arc") ("hyperref" "colorlinks" "citecolor=blue" "linkbordercolor={0 0 1}")))
   (TeX-run-style-hooks
    "latex2e"
    "amsart"
    "amsart11"
    "pstricks"
    "natbib"
    "xy"
    "enumerate"
    "mathrsfs"
    "amsmath"
    "amsthm"
    "amsfonts"
    "amssymb"
    "amsbsy"
    "graphicx"
    "enumitem"
    "physics"
    "bbm"
    "bm"
    "hyperref")
   (LaTeX-add-labels
    "sec:intro")
   (LaTeX-add-bibliographies
    "Zotero"))
 :latex)

