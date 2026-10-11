import gzip
import zlib

from spreadcore.discover import decode_frame


def test_decode_frames():
    assert decode_frame('{"a":1}') == ('{"a":1}', "text")
    assert decode_frame(b'{"a":1}') == ('{"a":1}', "utf8")
    assert decode_frame(gzip.compress(b'{"a":1}')) == ('{"a":1}', "gzip")
    c = zlib.compressobj(wbits=-15)
    assert decode_frame(c.compress(b'{"a":1}') + c.flush()) == ('{"a":1}', "deflate")
    assert decode_frame(b"\x82\xa1a\x01\xff")[1] == "base64"  # msgpack-like binary
