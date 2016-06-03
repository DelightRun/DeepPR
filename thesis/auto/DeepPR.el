(TeX-add-style-hook
 "DeepPR"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("XDUthesis" "WordOneHalf")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("natbib" "numbers" "sort&compress")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "ThesisFiles/Abstract"
    "ThesisFiles/Chapters"
    "ThesisFiles/Thanks"
    "ThesisFiles/Appendix"
    "XDUthesis"
    "XDUthesis10"
    "natbib")
   (LaTeX-add-bibliographies
    "ThesisFiles/RefFile"))
 :latex)

