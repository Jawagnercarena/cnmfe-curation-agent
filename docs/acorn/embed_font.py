"""
Subset Barlow Semi Condensed SemiBold to the glyphs the schematic uses,
convert to WOFF, base64-inline it as an @font-face named "ACORN Display",
and wire it into acorn-schematic.html as the primary --disp face.

Barlow (c) The Barlow Project Authors, SIL Open Font License 1.1.
The face is renamed on embed ("ACORN Display") per OFL reserved-font-name rules.
"""
import base64, io
from fontTools import subset

SRC  = "BarlowSemiCondensed-SemiBold.ttf"
HTML = "acorn-schematic.html"

# characters the display face has to cover (Basic Latin + the special dashes/dots/quotes used)
uni = set(range(0x20, 0x7F))
uni |= {0x00A0, 0x00B7, 0x2009, 0x2011, 0x2013, 0x2014,
        0x2018, 0x2019, 0x201C, 0x201D, 0x2026, 0x2192, 0x2212}

opt = subset.Options()
opt.flavor = "woff"
opt.desubroutinize = True
opt.notdef_outline = True
opt.layout_features = ["*"]      # keep kerning
opt.name_IDs = ["*"]             # keep the OFL copyright / license strings in the name table
opt.name_legacy = True
opt.glyph_names = False
opt.recalc_timestamp = False

font = subset.load_font(SRC, opt)
s = subset.Subsetter(options=opt)
s.populate(unicodes=uni)
s.subset(font)

buf = io.BytesIO()
subset.save_font(font, buf, opt)
data = buf.getvalue()
b64 = base64.b64encode(data).decode("ascii")
print(f"woff bytes: {len(data):,}   base64 chars: {len(b64):,}")

face = ('@font-face{font-family:"ACORN Display";font-style:normal;'
        'font-weight:400 900;font-display:swap;'
        f'src:url(data:font/woff;base64,{b64}) format("woff");}}')
credit = ('<!-- Display face: Barlow Semi Condensed (SIL OFL 1.1), '
          '(c) The Barlow Project Authors; subset & embedded as "ACORN Display". -->')

html = open(HTML, encoding="utf-8").read()
if "ACORN Display" in html:
    raise SystemExit("already embedded - aborting to avoid duplicate")

html = html.replace("<style>\n", credit + "\n<style>\n  " + face + "\n", 1)
html = html.replace('--disp:"Bahnschrift"', '--disp:"ACORN Display","Bahnschrift"', 1)

open(HTML, "w", encoding="utf-8").write(html)
print("injected @font-face and updated --disp in", HTML)
