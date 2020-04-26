%%
%% This is file `sareport.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% sareport.dtx  (with options: `class')
%% 
%% This is a generated file.
%% 
%% Copyright (C) 2017-2018 by imec-DistriNet-KU Leuven
%% 
%% This file may be distributed and/or modified under the conditions of
%% the LaTeX Project Public License, either version 1.3c of this license
%% or (at your option) any later version.  The latest version of this
%% license is in:
%% 
%%    http://www.latex-project.org/lppl.txt
%% 
%% and version 1.3c or later is part of all distributions of LaTeX version
%% 2005/12/01 or later.
%% 
\NeedsTeXFormat{LaTeX2e}[2013/03/31]
\ProvidesClass{res/theme}[]
\RequirePackage{ifthen}
\newboolean{anonymize}
\DeclareOption{peerreview}{\setboolean{anonymize}{true}}
\PassOptionsToClass{10pt,a4paper}{article}
\PassOptionsToPackage{english}{babel}
\PassOptionsToPackage{utf8}{inputenc}
\PassOptionsToPackage{T1}{fontenc}
\PassOptionsToPackage{a4paper}{geometry}
\PassOptionsToPackage{usenames,dvipsnames,svgnames,table}{xcolor}
\PassOptionsToPackage{inline}{enumitem}
\PassOptionsToPackage{first=0,last=1000}{lcg}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass{article}
\usepackage[numbered,autolinebreaks,useliterate]{mcode}
\lstset{xleftmargin=.25in} 
%\usepackage[framed,numbered,autolinebreaks,useliterate]{mcode}
%\usepackage[cachedir=minted]{minted}
\RequirePackage{babel}
\RequirePackage{inputenc}
\RequirePackage{fontenc}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{todonotes}
\RequirePackage{xparse}
\RequirePackage{expl3}
\RequirePackage{enumitem}
\RequirePackage{minitoc}
%\IfFileExists{fontawesome.sty}{%
%    \RequirePackage{fontawesome}
%}{%
%    \ClassWarning{theme}{You do not have fontawesome.sty available.
%    \MessageBreak You are recommended to upgrade your LaTeX distribution.
%    }
%    \newcommand{\faExclamationTriangle}{!}
%    \newcommand{\faInfoCircle}{?}
%    \newcommand{\faCheck}{v}
%    \newcommand{\faClipboard}{}
%}
\RequirePackage{advdate}
\RequirePackage{pdftexcmds,xparse}
\RequirePackage{totcount}
\RequirePackage{lcg}
\pagenumbering{arabic}

% MINTED
\usepackage[newfloat,cachedir=minted]{minted}
\usepackage{caption}
\newenvironment{code}{\captionsetup{type=listing}}{}
\SetupFloatingEnvironment{listing}{name=Code snippet}

\newcommand{\fakesection}[2]{
	\section*{#1#2}
	\par\refstepcounter{section}
	\addcontentsline{toc}{section}{#1}
}
\newcommand{\fakesubsection}[2]{
	\subsection*{#1#2}
	\par\refstepcounter{subsection}
	\addcontentsline{toc}{subsection}{#1}
}



\newtoks\authors
\newtoks\academicyear
\newtoks\casename
\newtoks\phasename
\newtoks\subtitle
\newtoks\groupname
\newtoks\course
\newtoks\department
\ifnum\month<10 \AdvYear{1} \fi
\academicyear{\AdvYear{-1}\the\year\AdvYear{1} -- \the\year}
\ifnum\month<10 \AdvYear{-1} \fi
\casename{{\color{red}Set \\casename!}}
\phasename{}
\subtitle{{\color{red} Set \\subtitle!}}
\department{Department of\\Computer Science}
\ExplSyntaxOn
\cs_set_eq:Nc \sareport_strcmp:nn { pdf@strcmp }
\seq_new:N \g_sareport_author_list_seq
\int_gzero_new:N \g_author_size_int
\NewDocumentCommand{\addAuthor}{mmm}
{
    \IfNoValueTF{#3}
    {
    }
    {
   \int_gincr:N \g_author_size_int
\ifthenelse{\equal{#3}{}}{%
    \sareport_add_author:nnnN { #1 } { #2 } {  } \g_author_size_int
}{%
            \sareport_add_author:nnnN { #1 } { #2 } { (#3) } \g_author_size_int
}
    }
}
\int_new:N \l_author_count_int
\tl_new:N \l_author_idx_tl
\NewDocumentCommand{\sortAuthors}{ }
 {
  \sareport_sort_authors:
  \int_zero:N \l_author_count_int
  \int_incr:N \l_author_count_int
  \int_do_while:nn { \l_author_count_int <= \g_author_size_int }
   {
     \seq_gpop_left:NNTF \g_sareport_author_list_seq \l_author_idx_tl
     {
         \IfEmpty{\groupname} {%
             \edef\tmp@groupname{\prop_item:cn {g_sareport_author_ \l_author_idx_tl _prop} { lnamestripped }}
             \expandafter\groupname{\tmp@groupname}
         } {%
             \edef\tmptwo@groupname{--\prop_item:cn {g_sareport_author_ \l_author_idx_tl _prop} { lnamestripped }}
             \groupname=\expandafter{\the\expandafter\groupname \tmptwo@groupname}
         }
         \IfEmpty{\authors} {%
             \edef\tmp@authors{%
                 \prop_item:cn {g_sareport_author_ \l_author_idx_tl _prop} { fname }  \noexpand &
                 \noexpand\textsc{\prop_item:cn {g_sareport_author_ \l_author_idx_tl _prop} { lname }} ~
                 \prop_item:cn {g_sareport_author_ \l_author_idx_tl _prop} { nbr }
             }
             \expandafter\authors{\tmp@authors}
         } {%
             \edef\tmptwo@authors{\noexpand\\
                 \prop_item:cn {g_sareport_author_ \l_author_idx_tl _prop} { fname }  \noexpand &
                 \noexpand\textsc{\prop_item:cn {g_sareport_author_ \l_author_idx_tl _prop} { lname }}  ~
                 \prop_item:cn {g_sareport_author_ \l_author_idx_tl _prop} { nbr }
             }
             \authors=\expandafter{\the\expandafter\authors \tmptwo@authors}
         }
      %\prop_show:c { g_sareport_author_ \l_author_idx_tl _prop }
     } { }
     \seq_gput_right:NV \g_sareport_author_list_seq { \l_author_idx_tl }
     \int_incr:N \l_author_count_int
    %\insertdata { \int_use:N \l_tmpa_int }
    %\int_incr:N \l_tmpa_int
   }
 }

\tl_new:N \l_tmpid_int
\cs_new_protected:Npn \sareport_add_author:nnnN #1 #2 #3 #4
 {
   \tl_set:NV \l_tmpid_int { #4 }
  \seq_gput_right:NV \g_sareport_author_list_seq { \l_tmpid_int }
  \prop_new:c { g_sareport_author_ \l_tmpid_int _prop }
  \prop_gput:cno { g_sareport_author_ \l_tmpid_int _prop } { sort } { \tl_lower_case:n {#2}\space\space\tl_lower_case:n {#1} }
  \prop_gput:cnn { g_sareport_author_ \l_tmpid_int _prop } { fname } { #1 }
  \prop_gput:cnn { g_sareport_author_ \l_tmpid_int _prop } { lname } { #2 }
  \edef\@tmp{#2}
  \prop_gput:cnx { g_sareport_author_ \l_tmpid_int _prop } { lnamestripped } { \expandafter\expandafter\expandafter\zap@space\expandafter\@tmp \space\@empty }
  \prop_gput:cnn { g_sareport_author_ \l_tmpid_int _prop } { nbr } { #3 }
  \prop_gput:cnx { g_sareport_author_ \l_tmpid_int _prop } { id } { \l_tmpid_int }
 }
\cs_new_protected:Npn \sareport_sort_authors:
 {
  \seq_gsort:Nn \g_sareport_author_list_seq
   {
    \prop_get:cnN {g_sareport_author_ ##1 _prop} { sort } \tl_first
    \prop_get:cnN {g_sareport_author_ ##2 _prop} { sort } \tl_snd
    \sareport_string_compare:NnNTF \tl_first {>} \tl_snd {\sort_return_swapped:} {\sort_return_same:}
   }
 }
\prg_new_conditional:Npnn \sareport_string_compare:NnN #1 #2 #3 {TF}
  {
   \if_int_compare:w \sareport_strcmp:nn {#1}{#3} #2 \c_zero
    \prg_return_true:
   \else:
    \prg_return_false:
   \fi
  }
\ExplSyntaxOff
\def\@makechapterhead#1{%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \Huge\bfseries \thechapter.
    \fi
    \interlinepenalty\@M
    \Huge \bfseries #1\par\nobreak
    \vskip 40\p@
  }}
\def\@makeschapterhead#1{%
  {\parindent \z@ \raggedright \normalfont
    \interlinepenalty\@M
    \Huge \bfseries #1\par\nobreak
    \vskip 40\p@
  }}
\geometry{tmargin=2.2cm, bmargin=2.2cm, lmargin=2cm, rmargin=2cm}
\rand
\newtotcounter{InstructionCounter}
\def\IfEmpty#1#2#3{%
  \edef\1{\the#1}%
  \ifx\1\empty%
    \expandafter\@firstoftwo%
  \else%
    \expandafter\@secondoftwo%
  \fi%
  {#2}{#3}%
}
\newcommand{\warning}[2][]{%
    \stepcounter{InstructionCounter}%
    \todo[inline, caption={2do},color=red!30, #1]{%
        \noindent\makebox[\textwidth][c]{%
            \begin{minipage}{0.975\textwidth}%
                \textbf{\faExclamationTriangle{} Attention:} #2%
            \end{minipage}%
        }%
    }%
}
\newcommand\note[2][]{%
    \stepcounter{InstructionCounter}%
    \todo[inline, caption={2do},color=orange!20, #1]{%
        \noindent\makebox[\textwidth][c]{%
            \begin{minipage}{0.975\textwidth}
                \textbf{\faInfoCircle{} Note:} #2%
            \end{minipage}%
        }%
    }%
}
\newcommand\hint[2][]{%
    \stepcounter{InstructionCounter}%
    \todo[inline, caption={2do},color=green!20, #1]{%
        \noindent\makebox[\textwidth][c]{%
            \begin{minipage}{0.975\textwidth}
                \textbf{\faCheck{} Hint:} #2%
            \end{minipage}%
        }%
    }%
}
\newcommand{\todoinline}[2][]{%
    \stepcounter{InstructionCounter}%
    \todo[inline, caption={2do},color=blue!20, #1]{%
        \noindent\makebox[\textwidth][c]{%
            \begin{minipage}{0.975\textwidth}
                \textbf{\faClipboard{} TODO:} #2%
            \end{minipage}%
        }%
    }%
}
\newcommand{\captioninstruction}{
    \hint{Add any essential information, necessary for interpreting
        the figure, in the caption. Be sure to add a separate short title
        for inclusion in the list of figures:
        \texttt{\textbackslash caption[shorttitle]\{longtitle\}}.\\
        If your explanation becomes too long for the caption, you can
        create a separate subsection. Don't forget to refer to the figure
        and vice versa.
    }%
}
\newcommand{\showdecisionsnotes}{
    \smallskip
    \note{This section discusses \emph{all} your architectural decisions
        \emph{in-depth}.
         First, \emph{all} decisions related to the non-functionals are
        discussed in detail. Next, \emph{all} other decisions are listed
        and discussed.
    }
    \smallskip
    \hint{Don't just say \emph{what} you have done.
        Explain \emph{why} you have done it.
    }
    \smallskip
}
\newcommand{\showcsnotes}{
    \smallskip
    \hint{No need to just repeat what we can see on the diagram.

        Don't do this: \emph{As you can see on fig. x: comp A consists
        of B and C, and C connects to D}.

        But, please do explain if there is anything non-trivial (e.g., a
        custom mapping from actors to external components on the
        context diagram).
    }
    \smallskip
    \captioninstruction{}
    \smallskip
    \hint{If you have any doubts about the size of your figures, it is
        better to make your figure too large than too small. Alternatively,
        you can test the readability by printing it.
    }
    \smallskip
    \warning{With regard to the context diagram, recall the lectures on
        what it means and should contain. Be sure not to miss any elements
        here. This is a frequent source of errors.
    }
    \smallskip
    \warning{Make sure your main component-and-connector and context
        diagrams are consistent.
    }
    \smallskip
}
\newcommand{\showdecompnotes}{
    \smallskip
    \hint{No need to just repeat what we can see on the diagram.

        Don't do this: \emph{As you can see on fig. x: comp A consists
        of B and C, and C connects to D}.

        But, please do explain if there is anything non-trivial
        (e.g., a custom mapping from actors to external components on
        the context diagram).
    }
    \smallskip
    \captioninstruction
    \smallskip
    \warning{\emph{Consistency between views!} Be sure to check for
        consistency between the client-server view and your
        decompositions.
    }
    \smallskip
    \warning{\emph{Consistency of a single decomposition!} Make sure that
        every interface provided or required by the decomposed component,
        is provided or required by a subcomponent in the decomposition.
    }
    \smallskip
}
\newcommand{\showdeploynotes}{
    \smallskip
    \hint{No need to just repeat what we can see on the diagram.

        Don't do this: \emph{As you can see on fig. x: components A and B
        are deployed on node C}.

        But, please do explain if there is anything non-trivial (e.g., a
        custom mapping from actors to external components on the context
        diagram).
    }
    \smallskip
    \captioninstruction
    \smallskip
    \warning{Connect nodes on the deployment diagram, \emph{not}
        components.
    }
    \smallskip
    \warning{\emph{Consistency between views!} Be sure to check for
        consistency between the client-server/decomposition view and your
        deployment view.
    }
    \smallskip
}
\newcommand{\showscenariosnotes}{
    \smallskip
    \hint{No need to just repeat what we can see on the diagram.

        Don't do this: \emph{As you can see on fig. x: component A calls
        operation b, next component C calls operation d}.

        But, please do explain if there is anything non-trivial (e.g., a
        custom mapping from actors to external components on the context
        diagram).
    }
    \smallskip
    \captioninstruction
    \smallskip
    \warning{Do include a list of which sequence diagrams together
        illustrate a which scenario from the assignment.
    }
    \smallskip
    \hint{Don't only model the `happy path' in your sequence diagrams.
        Take into account the quality attributes. For example, what
        happens when a certain component fails (Av) or overloads (P)?
        Use the sequence diagrams to illustrate how you have achieved
        the qualities in your architecture.
    }
    \smallskip
}
\newcommand{\showcatalognotes}{
    \smallskip
    \hint{Make sure the elements are sorted alphabetically.
        You can use the \texttt{\textbackslash componentItem\{name\}
        \{contents\}} command for this in your report.
        Note that you cannot use newlines in the componentItem content,
        but you can use \textbackslash\textbackslash.
    }
    \smallskip
    \hint{Common interfaces such as, for example, ping can be described
        separately so you don't have to repeat them for each component
        that provides it.
    }
    \smallskip
    \hint{Similarly, you can describe the exceptions separately as well,
        so you don't have to repeat what they mean for each operation
        that can throw them.
    }
    \smallskip
    \warning{Don't forget to include the exceptions in the method
        signature in the element catalog!
    }
    \smallskip
    \warning{Interfaces are uniquely identified by their name, regardless
        of the context (e.g., the component that provides it).
        In other words, two interfaces with the same name are considered
        identical.
    }
    \smallskip
    \warning{Don't forget to document the external interfaces!}
    \smallskip
}
\ExplSyntaxOn
\NewDocumentCommand{\componentItem}{mm}
{
    \seq_put_right:Nn \l_sareport_comps_seq {#1}
    \cs_new:cpn { sareport_comp_#1: } {
        \subsection{#1}#2
    }
}
\NewDocumentCommand{\printComponents}{ }
{
    \seq_sort:Nn \l_sareport_comps_seq
    {
        \string_compare:nnnTF {##1} {>} {##2} {\sort_return_swapped:} {\sort_return_same:}
    }
    \seq_map_inline:Nn \l_sareport_comps_seq { \use:c { sareport_comp_##1: } }
}
\seq_new:N \l_sareport_comps_seq
\prg_new_conditional:Npnn \string_compare:nnn #1 #2 #3 {TF}
{
    \if_int_compare:w \pdftex_strcmp:D {#1}{#3} #2 \c_zero
    \prg_return_true:
    \else:
    \prg_return_false:
    \fi
}
\ExplSyntaxOff
\newcommand{\sareport@shownoteswarning}{%
    \todo[inline, caption={2do},color=red!30]{%
        \noindent\makebox[\textwidth][c]{%
            \begin{minipage}{0.975\textwidth}%
                \textbf{\faExclamationTriangle{} Attention:}
                Your report still contains
                \total{InstructionCounter} instructions or
                comments.
                Make sure to delete all of them for the final
                version.
                (Run \texttt{pdflatex} at least two times
                after removing them to make sure this warning
                disappears.)
            \end{minipage}%
        }%
    }%
}%
\definecolor{kuleng}{RGB}{13,112,183}
\renewcommand{\maketitle}{%
    \pagenumbering{Alph}
    \begin{titlepage}
        \newpage
        \thispagestyle{empty}
        \frenchspacing
        %\hspace{-0.2cm}
        \vspace*{-4em}%
        \noindent%
        \IfFileExists{KULeuven.pdf}{%
            %\includegraphics[width=0.33\textwidth]{KULeuven}
        }{%
            \ClassWarning{sareport}{The file KULeuven.pdf is missing.}
        }

%        \smallskip\noindent
%            \fcolorbox{white}{kuleng}{
%                \parbox{0.313\textwidth}{\color{white}
%                    \centering
%                    \sffamily \bfseries
%                    \MakeUppercase{\the\department}}}
%            \smallskip
%        \hspace{\stretch{1}}
        \vspace*{3cm}
        %\vfill
        \begin{center}
            \begin{minipage}[t]{\textwidth}
                \begin{center}
                    {\huge {\rmfamily \the\casename}\\
                    {\Large \the\subtitle}\\
                  }%  \the\phasename\\}
                    \vspace{0.5cm}
                    \ifthenelse{\boolean{anonymize}}{%
                        \large{\textsc{Anonymized}}\\
                        \textsc{ \pdfmdfivesum{\the\groupname\the\year
                            \expandafter\arabic{rand}}}%
                    }{%
                        \large{\textsc{\the\groupname}}%
                    }
                    \end{center}
                \end{minipage}
            \vspace*{2cm}
            {\ifnum\totvalue{InstructionCounter}>0%
                \sareport@shownoteswarning{}%
            \fi}
        \end{center}
        \vfill
        \noindent
        \begin{minipage}[b][][t]{0.5\textwidth}
            \begin{flushleft}
            {\scshape Academic Year \the\academicyear}\\
            \medskip
            {\scshape
                \the\course
            }
            \medskip
            \end{flushleft}
        \end{minipage}
        \begin{minipage}[b][][b]{0.5\textwidth}
            \ifthenelse{\boolean{anonymize}}{%
                    {\textbf{Anonymized}}\\
                    {Academic year \the\academicyear}
                }{%
                    { \bfseries%
                    \begin{tabular}{>{\raggedleft}p{0.33\textwidth}@{~}p{0.66\textwidth}}
                        \the\authors
                    \end{tabular} %
                    }
                    {\bfseries \the\authors}
                }
        \end{minipage}
    \end{titlepage}
    \pagenumbering{arabic}


\sortAuthors
\fakelistoffigures
}
\endinput
%%
%% END OF BASE

\RequirePackage{tabu,longtable,booktabs,array}
\def\changemargin#1#2{\list{}{\rightmargin#2\leftmargin#1}\item[]}
\let\endchangemargin=\endlist 
\newcommand{\block}[1]{\vspace{1.4cm}\begin{changemargin}{2.5cm}{2.5cm}\large\textit{#1}\end{changemargin}\vspace{0.6cm}}
\RequirePackage{microtype} % Better word wrapping
  \RequirePackage{color}
\RequirePackage{xcolor}


\RequirePackage[font={footnotesize}]{caption}
\RequirePackage[ruled,vlined]{algorithm2e}
\newenvironment{snippet}[1][htb]
  {\renewcommand{\algorithmcfname}{Code Snippet}% Update algorithm name
   \vspace{0.3cm}\begin{algorithm}[#1]%
  }{\end{algorithm}\vspace{0.3cm}}
\RequirePackage{array}
\RequirePackage{tabu}
\RequirePackage{subfig}
\RequirePackage{multirow}
\RequirePackage{float}
\RequirePackage[all]{hypcap}
\RequirePackage{etoolbox}
\RequirePackage{xfrac}
\RequirePackage{wrapfig}
\RequirePackage{lscape}
\RequirePackage{rotating}
\RequirePackage{epstopdf}
\RequirePackage{fancyvrb}
\RequirePackage{tikz}
\RequirePackage{amssymb}
\makeatletter
\patchcmd{\@BVerbatim}
  {\BVerbatim@font}
  {\BVerbatim@font\footnotesize}
  {}{}
\makeatother
