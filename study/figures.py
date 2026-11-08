"""Render the constructed example, interval comparison and precision regimes."""
from pathlib import Path
import concept
import intervals
import regimes


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'run/figures'
    concept.main()
    intervals.main(root/'intervals')
    regimes.main(root/'regimes')
