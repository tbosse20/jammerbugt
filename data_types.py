from enum import Enum

class Medias(Enum):
    marta_video = 'mp4'
    marta_images = 'jpg'

    @classmethod
    def has_member_key(cls, key):
        return key in cls.__members__
    # https://stackoverflow.com/a/66891151

class Seabeds(Enum):
    '''
    Seabed types given from data.geus.dk
    (Colors manual set using plt)
    '''
    DYND =          [56,  168,  0]
    DYNET_SAND =    [152, 230,  0]
    SAND =          [255, 255,  0]
    GRUS =          [255, 152,  0]
    MORAENE =       [120,  50,  0]
    LER =           [168, 112,  0]
    GRUNDFJELD =    [217,  52, 52]
    UNKNOWN =       [  0,   0,  0]
    UNLOCATED =     None

namespace = {
    'gml': 'http://www.opengis.net/gml',
    'ms': 'http://example.com/ms',
    'ns0': 'http://www.opengis.net/gml',
    'ns1': 'http://mapserver.gis.umn.edu/mapserver'
}