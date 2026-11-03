"""Instance annotation workflow: split each cutout into one alpha per object.

Bước 0 (measurements before the annotator app is built), CLI in cli.py:
  dataset_index  pair images with cutouts, fingerprint pixels, compare two PCs
  alpha_split    split an alpha into separated objects, no model needed
  pilot          run alpha_split on a sample and draw contact sheets
  decoder_bench  encode on the GPU PC, time the HQ-SAM decoder on the CPU PC
"""
