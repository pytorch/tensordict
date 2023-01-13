# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
DATADIR       = _data

ZIPOPTS       ?= -qo

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

download:
	# imagenet tutorial data
	wget -nv -N https://download.pytorch.org/tutorial/hymenoptera_data.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/hymenoptera_data.zip -d $(SOURCEDIR)/reference/generated/tutorials/data

docs:
	make download
	make html
