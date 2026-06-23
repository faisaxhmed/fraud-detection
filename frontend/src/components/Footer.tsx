import styles from './Footer.module.css';

/** Identical on every view, per the project brief. */
export function Footer() {
  return (
    <footer className={styles.footer}>
      <a
        className={styles.link}
        href="https://www.linkedin.com/in/faisa-ahmed-41768a214/"
        target="_blank"
        rel="noreferrer"
      >
        Faisa Ahmed
      </a>
      <span className={styles.separator}>·</span>
      <span>2026</span>
      <span className={styles.separator}>·</span>
      <a className={styles.link} href="https://github.com/faisaxhmed/fraud-detection" target="_blank" rel="noreferrer">
        GitHub
      </a>
    </footer>
  );
}
