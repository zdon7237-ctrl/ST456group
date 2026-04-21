# Overleaf 上传说明

把下面这些文件上传到 Overleaf：

- `main.tex`
- `references.bib`
- 如果你已经下载了 ICML 2026 官方样式，再一起上传：
  - `icml2026.sty`
  - `icml2026.bst`

说明：

- `main.tex` 已经是主文件命名，上传后通常可以直接编译。
- 现在这个版本做了兼容处理：
  - 如果没有上传 `icml2026.sty` / `icml2026.bst`，也可以先用普通文章样式编译出 PDF。
  - 一旦把 ICML 样式文件补上，它会自动切回 ICML 格式。
- 如果 Overleaf 没有自动识别主文件，就手动把 `main.tex` 设为 Main document。
- 这份稿子里仍有少量需要你手动补的占位：
  - 作者姓名
  - 邮箱
  - train / validation 样本数
  - qualitative examples
  - individual contributions

当前上传包路径：

- `/Users/coy/Desktop/CodeX/ST456group/docs/overleaf_upload`
