#!/bin/bash
pdfcrop --margins '5 -120 0 30' smo.pdf smo_neu.pdf
pdftk smo_neu.pdf cat 4 output pseudo_code.pdf
rm smo_neu.pdf
