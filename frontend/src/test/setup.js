import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

// React Testing Library doesn't auto-unmount between tests outside of
// Jest's global afterEach — do it explicitly so each test starts fresh.
afterEach(() => {
  cleanup();
});
