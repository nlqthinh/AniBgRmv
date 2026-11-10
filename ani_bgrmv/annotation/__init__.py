"""Instance annotation workflow: split each cutout into one alpha per object.

  dataset_index    pair images with cutouts, fingerprint pixels, compare two PCs
  alpha_split      separated regions of an alpha, no model needed
  partition        share an alpha among objects (regions + HQ-SAM clicks),
                   lineart-snapped, 256-level, adding up to the original alpha
  auto_split       propose HQ-SAM clicks for groups of touching objects
  overlay          colored object maps (sheets, previews, editor)
  pilot            run the split on a fixed sample and draw contact sheets
  decoder_bench    bước 0: encode on the GPU PC, time the decoder on the CPU PC
  package          owner PC: one package per annotator ID folder
  image_edit       annotator: the edit of one image (clicks, merge, paint, undo)
  workspace        annotator: images of a package, status and output files
  annotator_window annotator app (AniBGRmv_CPU.exe --annotator)
  output_dir       generated folders (never deletes a foreign folder)
  cli              command line for the owner tools (also inside the EXE)
"""
