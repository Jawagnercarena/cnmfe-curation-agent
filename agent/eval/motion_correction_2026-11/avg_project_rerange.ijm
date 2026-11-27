/*
 * avg_project_rerange.ijm -- ONE-OFF copy of D:/2p data/scripts/avg_project.ijm
 * (2026-11-24, for 091526-pnb99, whose recording starts with ~360 dark raw
 * frames). Identical except for the 8-bit step: the display range is taken from
 * slice <rangeSlice> (spec line 5, 1-based) instead of slice 1. Every frame is
 * kept, the dark start included, so frame numbers stay aligned with behaviour.
 *
 *   line 0  stage directory holding the ordered, zero-padded reg_tif links
 *   line 1  output .tif path
 *   line 2  averaging factor n
 *   line 3  expected frame count, or 0 to skip the check
 *   line 4  status file the driver reads back
 *   line 5  slice (1-based, after averaging) whose min..max sets the 8-bit range
 */

var report = "";
var statusFile = "";

function say(s) {
    report = report + s + "\n";
    File.saveString(report, statusFile);
}

spec = getArgument();
if (spec == "") exit("no spec file given");
lines = split(File.openAsString(spec), "\n");

stageDir = replace(lines[0], "\r", "");
outTif   = replace(lines[1], "\r", "");
n        = parseInt(replace(lines[2], "\r", ""));
expected = parseInt(replace(lines[3], "\r", ""));
statusFile = replace(lines[4], "\r", "");
rangeSlice = parseInt(replace(lines[5], "\r", ""));

say("STAGE=begin heap_gb=" + d2s(IJ.maxMemory() / 1024 / 1024 / 1024, 1) + " range_slice=" + rangeSlice);

File.openSequence(stageDir);
if (nImages == 0) exit("File.openSequence opened nothing from " + stageDir);

fullID = getImageID();
total  = nSlices;
say("STAGE=imported frames=" + total + " bitdepth=" + bitDepth() +
    " width=" + getWidth() + " height=" + getHeight());

if (expected > 0 && total != expected) {
    say("WARN=frame count " + total + " does not match expected " + expected);
}
if (total < n) exit("only " + total + " frames, need at least n=" + n);

keep = floor(total / n) * n;
if (keep != total) {
    run("Slice Keeper", "first=1 last=" + keep + " increment=1");
    keptID = getImageID();
    selectImage(fullID);
    close();
    selectImage(keptID);
    say("STAGE=slicekeeper kept=" + keep + " dropped=" + (total - keep));
} else {
    say("STAGE=slicekeeper skipped (already divisible by " + n + ")");
}

inID = getImageID();
run("Grouped Z Project...", "projection=[Average Intensity] group=" + n);
avgID = getImageID();
selectImage(inID);
close();
selectImage(avgID);
say("STAGE=grouped_z frames=" + nSlices + " bitdepth=" + bitDepth());

// ---- 8-bit: the ONLY change from avg_project.ijm --------------------------
getMinAndMax(dmin0, dmax0);
if (rangeSlice < 1 || rangeSlice > nSlices) exit("range slice " + rangeSlice + " outside 1.." + nSlices);
setSlice(rangeSlice);
getStatistics(area, smean, smin, smax);
setMinAndMax(smin, smax);
getMinAndMax(dmin, dmax);
say("STAGE=pre8bit default_range=" + dmin0 + ".." + dmax0 + " range_from_slice=" + rangeSlice +
    " slice_mean=" + smean + " display_range=" + dmin + ".." + dmax);
run("8-bit");
say("STAGE=8bit frames=" + nSlices + " bitdepth=" + bitDepth());

saveAs("Tiff", outTif);
if (!File.exists(outTif)) exit("save produced no file at " + outTif);

say("STAGE=saved bytes=" + File.length(outTif));
say("STATUS=OK frames_in=" + total + " frames_kept=" + keep +
    " frames_out=" + nSlices + " factor=" + n);
