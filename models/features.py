""" Extract (traditional) linguistic features """
"""
from syltippy import syllabize

>>> syllables, stress = syllabize(u'supercalifragilísticoespialidoso')
>>> print(u'-'.join(s if stress != i else s.upper() for (i, s) in enumerate(syllables)))
su-per-ca-li-fra-gi-lis-ti-co-es-pia-li-DO-so

Hernández-Figueroa, Z; Rodríguez-Rodríguez, G; Carreras-Riudavets, F (2012). Separador de sílabas del español – Silabeador TIP. Available at https://tulengua.es
"""