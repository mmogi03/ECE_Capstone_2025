#!/usr/bin/env bash
set -e

# 1. Make a directory for global npm installs in your home
mkdir -p "${HOME}/.npm-global"

# 2. Configure npm to use that directory
npm config set prefix "${HOME}/.npm-global"

# 3. Ensure your shell loads the new PATH; detect your shell profile file
#    This works for bash (~/.bashrc) or Zsh (~/.zshrc)
PROFILE=""
if [ -n "$ZSH_VERSION" ]; then
  PROFILE="${HOME}/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
  PROFILE="${HOME}/.bashrc"
else
  PROFILE="${HOME}/.profile"
fi

# 4. Add the npm-global bin to your PATH if not already present
grep -qxF 'export PATH="$HOME/.npm-global/bin:$PATH"' "${PROFILE}" \
  || echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "${PROFILE}"

# 5. Reload your profile so PATH is updated now
#    (You may need to restart your shell instead.)
source "${PROFILE}"

# 6. Verify
echo "→ npm global prefix: $(npm config get prefix)"
echo "→ which npm:       $(which npm)"
echo "→ npm can now install globals without sudo!"

