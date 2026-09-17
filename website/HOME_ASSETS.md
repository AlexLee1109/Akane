# Home artwork

The Home page preserves the seated night hero and uses existing canonical artwork for the introduction and conversation. No hero artwork is repeated below the hero.

## Included

- Hero: `akane-night-hero.webp`, `akane-night-hero-1280.webp`, `akane-night-mobile.webp`; existing responsive artwork.
- Hero media card: existing `akane-night-thumb.webp`.
- Meet Akane: `akane-standing.webp`, optimized from the official `akane-hero.png` (full body, transparent background).
- Conversation: `akane-conversation.webp` and `akane-conversation-600.webp`, optimized from `akane-story.jpg` (alternate window portrait).
- Local runtime and footer: existing `akane-logo-192.png`.

## Artwork still needed

`HomeArtworkPlaceholder` in `src/pages/HomePage.tsx` provides intentional CSS atmosphere studies, not finished character illustrations. Each slot is named using `data-artwork-slot` and has a visible study label. Replace the placeholder with a responsive image when artwork is available, retaining the overlay copy.

| Slot | Suggested file | Brief | Suggested source size |
| --- | --- | --- | --- |
| `akane-morning` | `akane-morning.webp` | Akane beside a softly lit morning window; character above the lower text area. | 1000 × 1400 |
| `akane-desktop` | `akane-desktop.webp` | A quiet working desk with Akane visible in a desktop popup; blue dusk lighting. | 1000 × 1400 |
| `akane-evening` | `akane-evening.webp` | Akane in an evening room, relaxed alternate pose with city lights. | 1000 × 1400 |
| `akane-finale` | `akane-finale.webp` | Wide night skyline or balcony, small Akane figure at the right; open center for text, environment dominant. | 2400 × 1200 plus a mobile crop |

Preserve the official blue hair, blue eyes, white/blue outerwear, dark skirt, dark tie, and celestial details. Use the official full-body reference. Do not derive these scenes by repeating the hero.

Export optimized WebP/AVIF with explicit dimensions, responsive variants, and lazy loading. Reserve each existing scene's layout height. Remove the study label when actual artwork replaces it.
