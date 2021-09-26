# FISH data

__FISH data reference__ [Spatial organization of chromatin domains and compartments in single chromosomes](https://www.science.org/doi/full/10.1126/science.aaf8084)
* Legends for Tables S1 to S8 in the [Supplementary](https://www.science.org/doi/suppl/10.1126/science.aaf8084/suppl_file/wang-sm.pdf)
* Additional Data Table S1 ( [separate file](https://www.science.org/doi/suppl/10.1126/science.aaf8084/suppl_file/aaf8084_supportingfile_suppl1._excel_seq1_v1.xlsx) )
    > The genomic positions of the TADs probed in this work. The columns of the table are genome dataset name (hg18), chromosome name, ID numbers of the imaged TADs, start genomic coordinate of each TAD, and end genomic coordinate of each TAD. Genomic positions for different chromosomes are placed in different worksheets labeled with the names of the chromosomes.

* Additional Data Table S4 ([separate file](https://www.science.org/doi/suppl/10.1126/science.aaf8084/suppl_file/aaf8084_supportingfile_suppl1._excel_seq4_v1.xlsx))
    > Spatial coordinates of the central regions of imaged TADs for all 120 measured chromosomes for __Chr21__. The columns of the table are Serial numbers of the imaged chromosomes, ID numbers of the imaged TADs within the chromosomes, x positions of the TADs, y positions of the TADs, and z positions of the TADs. Unit for all position entries: µm. The positions are relative to the origin of the field of view (x, y) and the bottom of the z-stack (z). Occasionally, the fluorescence signals from the TADs were not sufficient to allow localization, giving rise to some empty cells in the table.
* Additional Data Table S5 ([separate file](https://www.science.org/doi/suppl/10.1126/science.aaf8084/suppl_file/aaf8084_supportingfile_suppl1._excel_seq5_v1.xlsx))
    > Spatial coordinates of the central regions of imaged TADs for all 151 measured chromosomes for __Chr22__. Columns are as defined in Table S4.
* Additional Data Table S6 ([separate file](https://www.science.org/doi/suppl/10.1126/science.aaf8084/suppl_file/aaf8084_supportingfile_suppl1._excel_seq6_v1.xlsx))
    > Spatial coordinates of the central regions of imaged TADs for all 111 measured chromosomes for __Chr20__. Columns are as defined in Table S4.
* Additional Data Table S7 ([separate file](https://www.science.org/doi/suppl/10.1126/science.aaf8084/suppl_file/aaf8084_supportingfile_suppl1._excel_seq7_v1.xlsx))
    > Spatial coordinates of the central regions of imaged TADs for all 95 measured chromosomes for __ChrXa__. Columns are as defined in Table S4.
* Additional Data Table S8 ([separate file](https://www.science.org/doi/suppl/10.1126/science.aaf8084/suppl_file/aaf8084_supportingfile_suppl1._excel_seq8_v1.xlsx))
    > Spatial coordinates of the central regions of imaged TADs for all 95 measured chromosomes for __ChrXi__. We used TAD coordinates obtained from the combined Hi-C data (8) of both Xa and Xi to determine labeling sites but note that the TAD structures are attenuated or absent on Xi (9, 35). Columns are as defined in Table S4.

__Locus Position__ Convert _hg18_ to _hg19_ by [Lift Genome Annotations](https://genome.ucsc.edu/cgi-bin/hgLiftOver)

Conversion failed:
  * Partially deleted __12th__ row (1-based) in __chr20	28080000	30880000__
  * Partially deleted __1st__ row (1-based) in __chr21	9880000	10240000__
  * Partially deleted __27th__ row (1-based) in __chr22	48560000	49691432__

__Validation method reference__ [Integrating Hi-C and FISH data for modeling of the 3D organization of chromosomes
](https://www.nature.com/articles/s41467-019-10005-6)