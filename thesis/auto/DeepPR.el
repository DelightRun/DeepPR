(TeX-add-style-hook
 "DeepPR"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("XDUthesis" "WordOneHalf")))
   (TeX-run-style-hooks
    "latex2e"
    "ThesisFiles/Abstract"
    "ThesisFiles/Chapters"
    "ThesisFiles/Thanks"
    "ThesisFiles/Appendix"
    "XDUthesis"
    "XDUthesis10")
   (LaTeX-add-bibliographies
    "ThesisFiles/RefFile"))
 :latex)

