%   Accuracy calculator from label data

format long

data = gTruth.LabelData.(1)
res = sum(data)/size(data, 1)
percent = res * 100

filter = {'*.txt', 'Text file'};

[file, path] = uiputfile(filter);

fid = fopen(strcat(path, '/', file), 'w');

fprintf(fid, 'Manual accuracy rating:\n\n');
fprintf(fid, 'Accuracy = %12.4f%%\n', percent);
fprintf(fid, '(%d correct frames out of %d total)\n\n', sum(data), size(data, 1))

fprintf(fid, 'By frame:\n\nCorrect\tFile\n');

for i = 1 : size(data, 1)
	imgdata = gTruth.DataSource.Source{i};
	[pathstr, name, ext] = fileparts(imgdata);
	fprintf(fid, '  %d\t\t%s\n', data(i), name);
end
	