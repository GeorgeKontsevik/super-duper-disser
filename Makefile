.PHONY: thesis-pdf thesis-clean

thesis-pdf:
	./scripts/render_itmo_thesis.sh

thesis-clean:
	rm -rf outputs/thesis/build outputs/thesis/thesis-itmo.pdf
