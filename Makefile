.PHONY: thesis-pdf thesis-chapter4-optimal-local thesis-clean

thesis-pdf:
	./scripts/render_itmo_thesis.sh

thesis-chapter4-optimal-local:
	./scripts/render_chapter4_optimal_local.sh

thesis-clean:
	rm -rf outputs/thesis/build outputs/thesis/thesis-itmo.pdf outputs/chapter4-optimal-local
