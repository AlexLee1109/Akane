# Website artwork and type

The website is complete using the existing official Akane artwork. All image references resolve under the configured Vite base path; no scene placeholders or visible study labels remain.

- Home hero: `akane-night-hero-1280.webp` / `akane-night-hero.webp`; small phones use `akane-night-mobile.webp`. High-priority eager loading, explicit dimensions, and a character-centered crop in a framed scene.
- Home conversation: `akane-conversation-600.webp`, lazy-loaded behind a clearly labeled written conversation example.
- About: responsive `akane-conversation-600.webp` / `akane-conversation.webp`, cropped inside an editorial frame.
- Demo: transparent `akane-standing.webp`, presented at character scale on desktop and as a compact portrait on mobile. This is static artwork with subtle CSS idle movement, not animated facial embodiment.
- Closing: `akane-standing.webp`, lazy-loaded in a separate portrait composition.
- Brand, welcome state, and messages: `akane-logo-192.png`.
- Existing source images, social artwork, and unused variants are retained.

Manrope is self-hosted as the 25KB variable Latin WOFF2 in `public/assets/manrope-latin.woff2`, preloaded from the configured base path. Its SIL Open Font License is included in `public/assets/manrope-OFL.txt`. Editorial italic headings use the system Georgia font.

Optional future artwork: an alternate seated conversational pose matching the official design, exported as a transparent WebP around 800 × 1200 with space around hair and sleeves, could add variety to the demo. The implemented site does not depend on it.
