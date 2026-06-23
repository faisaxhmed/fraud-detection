import styles from './Nav.module.css';

export type View = 'product' | 'about' | 'caseStudy';

interface Props {
  view: View;
  onNavigate: (view: View) => void;
}

/** Persistent top nav: the brand on the left always returns to the main product view, the
 * links on the right each switch to their own static view, independent of one another. */
export function Nav({ view, onNavigate }: Props) {
  return (
    <nav className={styles.nav}>
      <button type="button" className={styles.brand} onClick={() => onNavigate('product')}>
        <img src="/favicon.svg" alt="" className={styles.logo} />
        Explainable Fraud Detector
      </button>
      <div className={styles.links}>
        <button
          type="button"
          className={`${styles.link} ${view === 'about' ? styles.active : ''}`}
          onClick={() => onNavigate('about')}
        >
          About
        </button>
        <button
          type="button"
          className={`${styles.link} ${view === 'caseStudy' ? styles.active : ''}`}
          onClick={() => onNavigate('caseStudy')}
        >
          Case Study
        </button>
      </div>
    </nav>
  );
}
