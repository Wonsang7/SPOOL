package org.spool.fiji;

import java.io.InputStream;
import java.net.URL;
import java.security.CodeSource;
import java.util.Properties;

/** Reads metadata from the executing class's own artifact, never another JAR. */
public final class PluginBuildInfo {
    private static final String RESOURCE = "META-INF/spool-build.properties";

    private PluginBuildInfo() {}

    public static String describe(Class<?> executingClass) {
        String version = "unknown";
        String build = "unknown";
        String source = "unknown";
        try {
            CodeSource codeSource = executingClass.getProtectionDomain().getCodeSource();
            if (codeSource != null && codeSource.getLocation() != null) {
                URL location = codeSource.getLocation();
                source = location.toExternalForm();
                // Resolve against this code source rather than the class loader:
                // duplicate installations must not supply another copy's metadata.
                URL metadata = source.endsWith("/")
                        ? new URL(location, RESOURCE)
                        : new URL("jar:" + source + "!/" + RESOURCE);
                Properties properties = new Properties();
                try (InputStream input = metadata.openStream()) {
                    properties.load(input);
                }
                version = resolved(properties.getProperty("version"));
                build = resolved(properties.getProperty("build.id"));
            }
        } catch (Exception ignored) {
            // Diagnostics must not prevent reconstruction in development or
            // unusual class loaders. Never guess a version from the filename.
        }
        return "SPOOL Fiji: version=" + version + "; build=" + build + "; source=" + source;
    }

    private static String resolved(String value) {
        return value == null || value.trim().isEmpty() || value.contains("${")
                ? "unknown" : value.trim();
    }
}
