# orbits
Repository for orbit programs and stuff.

So I can stop wondering about tracking changes and allow for up to date version.

Note: For downloading latest file, click on `raw`, then `ctrl+s`. Make sure to select the right file type.

Otherwise, check under `Releases`

## Python
- orbits.py: Creates orbit class and space class for methods dealing with orbit instances. See documentation at the beginning of the file.

## Matlab
- orbit.m
- space.m

(orbit.m depends on space.m)

- orbits_demo.m: Example use of orbit.m and space.m.

Based off orbits.py, but with less features.

## Jupyter Notebooks
- Orbit notebook.ipynb: Used for development and debugging of orbits.py (backup file).
- Orbit calculations notebook.ipynb: Calculations for formulas implemented in orbits.py.
- Planet orbital elements.ipynb: Downloads data about planets and plans an interplanetary transfer.
- Anomalie moyenne - vraie.ipynb: Calculates the transformation from mean anomaly to true anomaly and formats it to be inputted in a LaTeX document.

# License
Making my code publicly available, I am distributing it under Mozilla Public License 2.0.

This means essentially that you are free to use, modify, redistribute this code, but the code and any modification to it must remain open source and under this same License or other compatible license; it is file-based and does not apply to private use. Copyright and license notices must be conserved. See [License](https://github.com/Alexandre867/orbits/blob/main/LICENSE) for the full license, or https://www.mozilla.org/en-US/MPL/2.0/FAQ/ to understand the permitted uses.
