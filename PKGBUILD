pkgname=bgm
pkgver=0.1.0
pkgrel=1
pkgdesc="Native wallpaper catalog and collection manager for Hyprland and wpaperd"
arch=('x86_64')
url="https://github.com/Markauto/RustBackgroundManager"
license=('MIT')
depends=('glibc' 'libgcc' 'hip-runtime-amd')
makedepends=('cargo')
options=('!debug' '!lto')
optdepends=(
    'kitty: native image previews in the TUI'
    'wpaperd: managed wallpaper collection integration'
    'xdg-utils: opening wallpapers from the TUI'
)

# This is a local-checkout package. Keeping makepkg's work tree separate is
# important because its default $srcdir would collide with the crate's src/.
if [[ $BUILDDIR == "$startdir" ]]; then
    BUILDDIR="$startdir/.makepkg"
fi
source=()
sha256sums=()

prepare() {
    cd "$startdir"
    export CARGO_HOME="$srcdir/cargo-home"
    cargo fetch --locked --target "$CARCH-unknown-linux-gnu"
}

build() {
    cd "$startdir"
    export CARGO_HOME="$srcdir/cargo-home"
    export CARGO_TARGET_DIR="$srcdir/cargo-target"
    export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--remap-path-prefix=$srcdir=/usr/src/$pkgname-$pkgver"
    cargo build --frozen --release --features rocm
}

check() {
    cd "$startdir"
    export CARGO_HOME="$srcdir/cargo-home"
    export CARGO_TARGET_DIR="$srcdir/cargo-target"
    export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--remap-path-prefix=$srcdir=/usr/src/$pkgname-$pkgver"
    cargo test --frozen --features rocm
}

package() {
    install -Dm755 "$srcdir/cargo-target/release/bgm" "$pkgdir/usr/bin/bgm"
    install -Dm644 "$startdir/README.md" "$pkgdir/usr/share/doc/$pkgname/README.md"
}
