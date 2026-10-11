package org.spool.fiji;

import java.io.InputStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;
import java.util.jar.Attributes;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.jar.JarOutputStream;
import org.junit.Test;
import static org.junit.Assert.*;

/** Runs after shading and checks the exact artifact used by the release job. */
public class PluginBuildInfoIT {
    private Path artifact() {
        return Paths.get(System.getProperty("spool.test.jar"));
    }

    @Test public void deployableJarMatchesMavenAndContainsGpuBridge() throws Exception {
        try (JarFile jar = new JarFile(artifact().toFile())) {
            Properties metadata = new Properties();
            try (InputStream input = jar.getInputStream(jar.getJarEntry("META-INF/spool-build.properties"))) {
                metadata.load(input);
            }
            Properties maven = new Properties();
            try (InputStream input = jar.getInputStream(jar.getJarEntry("META-INF/maven/org.spool/spool_fiji/pom.properties"))) {
                maven.load(input);
            }
            String version = System.getProperty("spool.test.version");
            String build = System.getProperty("spool.test.build.id");
            assertEquals(version, metadata.getProperty("version"));
            assertEquals(version, maven.getProperty("version"));
            Attributes manifest = jar.getManifest().getMainAttributes();
            assertEquals(version, manifest.getValue("Implementation-Version"));
            assertEquals(build, metadata.getProperty("build.id"));
            assertEquals(build, manifest.getValue("SPOOL-Build-Id"));
            assertFalse(version.contains("${"));
            assertFalse(build.contains("${"));
            assertNotNull(jar.getJarEntry("org/jocl/CL.class"));
            assertNotNull(jar.getJarEntry("org/spool/gpu/spool.cl"));
        }
    }

    @Test public void renamedUpdateSiteJarReportsItsOwnMetadataDespiteDuplicateResource() throws Exception {
        Path dir = Files.createTempDirectory("spool-version-test");
        Path renamed = dir.resolve("spool_fiji-0.1.2-SNAPSHOT.jar");
        Path stale = dir.resolve("stale.jar");
        try {
            Files.copy(artifact(), renamed);
            writeMetadataJar(stale, "version=stale\nbuild.id=stale\n");
            // Stale metadata is first on the classpath, but the executing class
            // comes from the renamed deployable JAR.
            try (URLClassLoader loader = new URLClassLoader(new URL[]{stale.toUri().toURL(), renamed.toUri().toURL()}, null)) {
                Class<?> info = loader.loadClass("org.spool.fiji.PluginBuildInfo");
                String log = (String) info.getMethod("describe", Class.class).invoke(null, info);
                assertEquals("SPOOL Fiji: version=" + System.getProperty("spool.test.version")
                        + "; build=" + System.getProperty("spool.test.build.id")
                        + "; source=" + renamed.toUri().toURL().toExternalForm(), log);
            }
        } finally {
            Files.deleteIfExists(stale);
            Files.deleteIfExists(renamed);
            Files.deleteIfExists(dir);
        }
    }

    @Test public void missingMetadataReportsUnknownWithoutThrowing() throws Exception {
        Path jar = Files.createTempFile("spool-missing-metadata", ".jar");
        try {
            try (JarOutputStream output = new JarOutputStream(Files.newOutputStream(jar));
                 InputStream input = PluginBuildInfo.class.getResourceAsStream("PluginBuildInfo.class")) {
                output.putNextEntry(new JarEntry("org/spool/fiji/PluginBuildInfo.class"));
                byte[] buffer = new byte[4096];
                int count;
                while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
                output.closeEntry();
            }
            try (URLClassLoader loader = new URLClassLoader(new URL[]{jar.toUri().toURL()}, null)) {
                Class<?> info = loader.loadClass("org.spool.fiji.PluginBuildInfo");
                String log = (String) info.getMethod("describe", Class.class).invoke(null, info);
                assertTrue(log.startsWith("SPOOL Fiji: version=unknown; build=unknown; source="));
            }
        } finally {
            Files.deleteIfExists(jar);
        }
    }

    private void writeMetadataJar(Path jar, String metadata) throws Exception {
        try (JarOutputStream output = new JarOutputStream(Files.newOutputStream(jar))) {
            output.putNextEntry(new JarEntry("META-INF/spool-build.properties"));
            output.write(metadata.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            output.closeEntry();
        }
    }
}
