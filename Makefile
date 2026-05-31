SHELL := /bin/bash

# Python executable
PY ?= python

# Inputs / outputs
INPUT_DIR ?= data/raw/images
OUT_DIR ?= outputs/sam2_prompted
WINDOWS_DIST_DIR ?= build/Heidelbeeren-Bewertung-App
WINDOWS_WIN_COPY_PARENT ?= /mnt/c/Users/LeifGarbe/Music
WINDOWS_WIN_COPY_DIR ?= $(WINDOWS_WIN_COPY_PARENT)/Heidelbeeren-Bewertung-App

# SAM 2.1 model + device
MODEL ?= facebook/sam2.1-hiera-large
DEVICE ?= cuda

# Extra flags (optional)
RUNNER_FLAGS ?=
VIZ_FLAGS ?=

.PHONY: all run_prompted viz_both viz_post viz_raw clean_visuals windows-dist help

# Full pipeline: run SAM 2.1 prompted, then visualize prompts on both variants
all: run_prompted viz_both

# Generate overlays with SAM 2.1 (prompted). Produces:
#  - $(OUT_DIR)/overlays, $(OUT_DIR)/colored (post-processed/selected)
#  - $(OUT_DIR)/overlays_raw, $(OUT_DIR)/colored_raw (ALL raw candidates, no post-processing)
run_prompted:
	$(PY) -m src.run_sam2_prompted \
		--input-dir "$(INPUT_DIR)" \
		--output-dir "$(OUT_DIR)" \
		--model-id "$(MODEL)" \
		--device "$(DEVICE)" \
		$(RUNNER_FLAGS)

# Visualize prompts on both variants in one go
viz_both:
	$(PY) -m src.visualize_prompt_points \
		--both \
		--overlays-dir "$(OUT_DIR)/overlays" \
		--originals-dir "$(INPUT_DIR)" \
		$(VIZ_FLAGS)

# Visualize prompts on post-processed overlays only
viz_post:
	$(PY) -m src.visualize_prompt_points \
		--overlays-dir "$(OUT_DIR)/overlays" \
		--originals-dir "$(INPUT_DIR)" \
		--write-dir "$(OUT_DIR)/overlays_with_prompts" \
		$(VIZ_FLAGS)

# Visualize prompts on raw overlays (no post-processing) only
viz_raw:
	$(PY) -m src.visualize_prompt_points \
		--overlays-dir "$(OUT_DIR)/overlays_raw" \
		--originals-dir "$(INPUT_DIR)" \
		--write-dir "$(OUT_DIR)/overlays_with_prompts_without_post_processing" \
		$(VIZ_FLAGS)

# Remove only the visualization outputs (keeps generated overlays)
clean_visuals:
	rm -rf "$(OUT_DIR)/overlays_with_prompts" "$(OUT_DIR)/overlays_with_prompts_without_post_processing"

# Package and verify the Windows GUI bundle
windows-dist:
	rm -rf "$(WINDOWS_DIST_DIR)"
	$(PY) -m tools.prepare_windows_app --dest "$(WINDOWS_DIST_DIR)"
	$(PY) -m tools.verify_windows_bundle --bundle "$(WINDOWS_DIST_DIR)"
	@if [ -d "$(WINDOWS_WIN_COPY_PARENT)" ]; then \
		echo "Kopiere Windows-App nach $(WINDOWS_WIN_COPY_DIR)"; \
		rm -rf "$(WINDOWS_WIN_COPY_DIR)"; \
		mkdir -p "$(WINDOWS_WIN_COPY_PARENT)"; \
		cp -a "$(WINDOWS_DIST_DIR)" "$(WINDOWS_WIN_COPY_DIR)"; \
	else \
		echo "Hinweis: Zielverzeichnis $(WINDOWS_WIN_COPY_PARENT) existiert nicht, Windows-Kopie wird uebersprungen."; \
	fi

help:
	@echo "make all         - Run SAM 2.1 prompted, then visualize both variants"
	@echo "make run_prompted - Generate overlays (and overlays_raw) with SAM 2.1"
	@echo "make viz_both     - Visualize prompts on overlays and overlays_raw"
	@echo "make viz_post     - Visualize prompts on post-processed overlays only"
	@echo "make viz_raw      - Visualize prompts on raw overlays only"
	@echo "make clean_visuals- Remove visualization outputs"
	@echo "make windows-dist - Build + prüfen der Windows-App (output: $(WINDOWS_DIST_DIR), Kopie nach C:\\Users\\LeifGarbe\\Music\\Heidelbeeren-Bewertung-App falls vorhanden)"
	@echo "Variables: INPUT_DIR, OUT_DIR, MODEL, DEVICE, RUNNER_FLAGS, VIZ_FLAGS"
