// Deployment-tunable model location.
//
// A same-origin ./models directory is right for a checkout served locally
// or a Pages deployment that bundled weights in with `--models-dir`
// (tools/build_site.py). A deployment that instead points at a pinned
// remote release asset (e.g. because GitHub Pages' 1 GB cap makes shipping
// weights inline impractical) rewrites this ONE line post-build --
// .github/workflows/pages.yml does exactly that via the
// AMINX_MODEL_BASE_URL repository variable. Nothing else in the site reads
// a model path from anywhere else.
export const MODEL_BASE = "./models";
