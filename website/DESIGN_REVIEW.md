# Akane website redesign

## Audit and design decisions

The supplied checkout already contained uncommitted frontend and backend work. This redesign builds on that checkout and changes only the website.

- The inherited homepage placed an absolutely positioned illustration behind the copy. The new hero uses an explicit two-column grid and a single framed character scene, with separate mobile composition.
- Small demo controls and a dark-on-dark conversation panel weakened readability. The rebuilt workspace pairs a midnight character stage with an ivory conversation surface, stronger message contrast, and visible connection labels.
- The mobile demo previously imposed a 600px minimum height. It now uses the dynamic viewport, a compact character strip, and a scrollable message history so the composer fits short screens.
- Shared tokens unify color, spacing, content widths, radii, buttons, and focus states. Self-hosted Manrope provides consistent interface typography; restrained Georgia italics give major headings an editorial accent. No third-party font requests or added application dependencies.
- Home tells four distinct stories: continuing companionship, selective memory, developing perspective, and local control. The conversation illustration is explicitly labeled as written example content.
- Technology presents an accessible architecture diagram, ordered request path, state responsibilities, runtime specification, guest isolation, and implemented/ongoing/future distinctions. Content follows current code where README descriptions are older: the checkout includes expanded developmental state and an event-driven coordinator. Mood, Relationship, and InnerLife are not presented as current canonical records.
- About uses editorial sections, project principles, an undated progression, and creator attribution without invented dates, quotations, or credentials.
- The existing client, session storage, health checks, retry schedule, generation cancellation, and guest lifecycle remain intact. HashRouter and the GitHub Pages workflow remain intact; routes use `/Akane/#/...` for direct static loading.

## Verification in this pass

- `VITE_BASE_PATH=/Akane/ npm run build`: passed, including TypeScript.
- `.venv/bin/python -m pytest -q`: 507 passed.
- `node --test tests/test_ttft_client.cjs`: 3 passed.
- `git diff --check -- website`: passed.
- Chromium: all four routes at 320, 390, 768, 1440, and 1920px; no horizontal overflow, broken images, or page errors. Home, Demo, Technology, and About screenshots inspected at desktop and mobile sizes.
- Keyboard navigation, mobile menu Escape dismissal, direct hash-route loading, skip link, and Technology section link checked.
- Intercepted API checks passed for health, guest creation, bearer-token handling, streamed response parsing/rendering, stop, reset, rate-limit feedback, session-expiry recovery, termination, offline state, preview isolation/clear, reconnect, and stored-session revalidation.
- Additional short-screen and tablet checks cover composer visibility, preview send, self-hosted font and all official artwork loading, menu dismissal after keyboard navigation, and reduced motion.

Browser API interception verifies frontend behavior, not the availability or response quality of a deployed model. Production publication was not performed. No additional artwork is required.
