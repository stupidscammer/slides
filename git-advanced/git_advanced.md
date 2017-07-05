name: inverse
layout: true
class: underscore
---
class: center, middle, hero

.title[
  # .red[git] collaborative workflow
  ## Alexandre Boucaud
]

.bottom[.small[http://www.ias.u-psud.fr/aboucaud/git-collab.html]]

???

Speaker notes go after ???

Toggle speaker view by pressing P

Pressing C clones the slideshow view,
which is the view to put on the projector
if you're using speaker view.

Press ? to toggle keyboard commands help.

---

## Foreword

Notions to be familiar with.red[*]

* basic workflow: `add`, `commit`, `push`
* branching: `branch`, `checkout`
* merging: `merge` with fast-forward and without

.footnote[.red[*] otherwise check out the [beginners tutorial][gitbeg]]

[gitbeg]: http://www.ias.u-psud.fr/pperso/aboucaud/git/git-tutorial-slides.html

---

# Agenda

1. [Collaborative tips & tools](#intro)
2. [Continuous integration](#ci)

---
name: intro
class: center, middle

# 1. Collaborative work with .red[git]


---
class: middle

## README.md

---

## Content of a good README

* Synopsis (elevator pitch)
* Code example (if library)
* Motivation / Rationale / How it differs from alternatives Installation
* Getting started
* API reference (or link to it) (if library)
* Cross-links to contribution guide / wiki / license(s)

---
class: middle

## CHANGELOG

---
## Keep the version history

.center[http://keepachangelog.com/]

---
class: middle

## Commit messages

---

## Lazy message are just useless

```
$ git log --oneline -5
dfb1fa4 Fixup
98e0832 Update
7848d7f New commit
c95fd42 Commit
f124acb First commit
```

A commit message = .red[quick doc] for others & for the future.

Others = .red[YOU] in 3 months.

---

## Good commit message #1

* .red[1st line:] < 50 chars & imperative present

```
Fix incorrect variable name
```

rather than

```
I fixed the name of the variable in the function since I decided
it was prettier that way
```

---

## Good commit message #2

If more details needed (justification or important changes):

* Blank line before details
* Details paragraph wrapped to < 72 chars

---

## Good commit message #3

* .red[Use Markdown syntax]
    * italics, bold
    * URL links
    * lists

* Auto-close issues on server using hooks (e.g. Fixes #42)


---
class: middle

## Rebasing

---

## Tip #1: interactive rebase should be a reflex

git motto: .red[commit often, push when needed].
=> .red[clean up] your log before pushing.

Review all commits since last `push`
```
git log --oneline --graph --decorate @{push}..
```
Use interactive rebase to reorder, rewrite, concatenate commits
```
git rebase -i
```

---

## Tip #2: `pull` default behavior is not your friend

* .red[pull] = .red[fetch] + .red[merge]
* Collaborators working on the same branch merge all the time upon itself.
this creates a .red[messy graph].

```
| *   af0d902 Merge branch 'datamodelPipeline' of http://euclid-git.roe.ac.uk/PF-MER/MER_Pipeline.git into datamodelPipeline
| |\
| | *   ad8735d Merge remote-tracking branch 'origin/develop' into datamodelPipeline
| | |\
| |_|/
|/| |
* | | 2012bfe Correct a bug in the transformation from a weight image to a flag image.
| * | ddc357e Implement some use of the data model items in the mosaicing step.
| |/
* | 057d87e Version to be merged with the development branch.
```

---

## Setting `pull.rebase` to `preserve`

```
git pull --rebase=preserve       # live
git config pull.rebase preserve  # configuration setting
```

**Idea**: rebase on top of the updated remote branch.
Keeps a .red[clean graph] with a single line per branch.

```
| |\
| | * c5dfa84 Fix zero point setter: getter method outputs a dictionary
| | * 723f9fd Fix temporary files path with tphot_convolve
| | *   69d0650 Merge branch 'kernel' into kernel
| | |\
| | * | 614c9ec Remove install_script instruction
| | * | 5069791 Fix bad indentation
```

---
class: middle
## Merge/pull request

---
class: middle
## LICENSE

---

## A note about Licenses

.red[No license ≠ “do what you want”]

Actually the opposite: “all rights reserved”

Learn more at choosealicense.com and mass-apply with addalicense.com


---
name: ci
class: center, middle

# 2. Continuous integration (CI)

---

## GitLab setup

* go to https://git.ias.u-psud.fr/

.center[<img src="img/gitlab_signin.png" width="80%" alt="gitlab" >]

* authenticate with LDAP to create your account
---

## Motivations


---

## GitLab CI

Continuous integration is fully integrated to GitLab

[Documentation](http://docs.gitlab.com/ee/ci/)
[Quick start guide](https://docs.gitlab.com/ce/ci/quick_start/)

---

## `.gitlab-ci.yml`



## Custom build image

```Dockerfile
*FROM debian:stretch

MAINTAINER Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

RUN apt-get update && \
    apt-get install -y tox \
*        python2.7 \
        python-pip python-pytest-cov python-coverage \
        python-numpy python-matplotlib \
        python-astropy python-photutils python-healpy python-wcsaxes \
*        python3.5 \
        python3-pip python3-pytest-cov python3-coverage \
        python3-numpy python3-matplotlib \
        python3-astropy python3-photutils python3-healpy python3-wcsaxes
```

---
name: tips
class: center, middle

# 6. Wrap up and tips

---

---

## Links

[Official documentation](https://git-scm.com/doc)

**Beginners**

* [Try git](https://try.github.io) - *interactive tutorial*

* [git - the simple guide](http://rogerdudler.github.io/git-guide/)

**Advanced users**

* [Atlassian tutorials](https://www.atlassian.com/git/tutorials)

* [Successful branching model](http://nvie.com/posts/a-successful-git-branching-model/)

---

class: center, middle, hero

.title[
  # Thank You
  ### [aboucaud@ias.u-psud.fr][abcd]
  ]

.footnote[
  .small[
  This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc]
  ]
[![](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)][cc]
]

[abcd]: mailto:alexandre.boucaud@ias.u-psud.fr
[cc]: http://creativecommons.org/licenses/by-sa/4.0
