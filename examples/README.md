Short explanation of the files in here.

We have two kinds of offline test data:

* "old" test data in the HZDR cloud (files `simData_bunches_012345.bp` etc.)
* "new" data called LWFA in HZDR's hemera cluster `/bigdata/hplsim/production/...`

For more details, see the collaborative notes (https://notes.desy.de/ask_us_for_the_link#Data).

Files using old data:

* `extract_particles.ipynb` (PIC simulation input, particle / "phase space" / point
  cloud data)
* `plot_particles.ipynb`
* `extract_radiation.ipynb` (PIC simulation outputs)

Files using new data:

* `LWFA_particle_data.ipynb`
* `LWFA_radiation_data.ipynb`
