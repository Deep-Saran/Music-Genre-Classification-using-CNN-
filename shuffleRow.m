function ret = shuffleRow(mat)
[r c] = size(mat);
shuffledRow = randperm(r);
ret = mat(shuffledRow, :);