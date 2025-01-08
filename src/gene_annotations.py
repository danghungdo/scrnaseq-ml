from typing import List

G1_CYCLE = [
    "CDKN1A",
    "CDKN1B",
    "CDKN1C",
]

ERYTHROID = [
    "PTPN1",
]

PIONEER_FACTORS = [
    "FOXA3",
    "HOXA13",
    "HOXC13",
    "TP73",
    "MIDN",
    "HOXC13",
]

GRANULOCYTE_APOPTOSIS = [
    "SPI1",
    "CEBPA",
    "CEBPB",
    "CEBPE",
]

PRO_GROWTH = [
    "KLF1",
    "ELMSAN1",
    "MAP2K3",
    "MAP2K6",
]

MEGAKARYOCYTE = [
    "MAPK1",
    "ETS2",
]

PROGRAMMES = {
    "G1 cell cycle": G1_CYCLE,
    "Erythroid": ERYTHROID,
    "Pioneer factors": PIONEER_FACTORS,
    "Granulocyte apoptosis": GRANULOCYTE_APOPTOSIS,
    "Pro-growth": PRO_GROWTH,
    "Megakaryocyte": MEGAKARYOCYTE,
}

def annotate_genes(genes: List) -> List:
    gene_programme = []
    for target_pert in genes:
        if target_pert == "ctrl":
            gene_programme.append("Control")
            continue

        found_programme = False
        for programme, pert_list in PROGRAMMES.items():
            for pert in pert_list:
                if pert == target_pert:
                    gene_programme.append(programme)
                    found_programme = True
                    break

        if not found_programme:
            gene_programme.append("Unknown")
    return gene_programme